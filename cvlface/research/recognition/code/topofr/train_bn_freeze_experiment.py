#!/usr/bin/env python3
"""
BN 冻结对比实验
=================
对比冻结 BN 和不冻结 BN 对预训练模型的影响

实验设计:
1. 加载预训练模型，直接评估 agedb_30 (baseline)
2. 冻结 BN，训练到 10%, 20%, 30%, 40%, 50% 进度时评估
3. 重新加载预训练模型，不冻结 BN，训练到相同进度时评估
4. 对比两组结果
"""

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
import pandas as pd
import torch
import config
from config import Config
from models import get_model
from classifiers import get_classifier
from losses import get_margin_loss
from dataset import get_train_dataset, set_epoch
from evaluations import get_evaluator_by_name
from general_utils import random_utils
from optims.optims import make_optimizer
from lightning.fabric.loggers import CSVLogger
from optims.lr_scheduler import make_scheduler, scheduler_step, get_last_lr
from pipelines import pipeline_from_config
import omegaconf
from tqdm import tqdm
import time
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
import datetime
from pefts import apply_peft
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset
import json


def freeze_bn_layers(model):
    """冻结所有 BN 层的统计量和参数"""
    frozen_count = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            module.eval()  # 固定统计量
            for param in module.parameters():
                param.requires_grad = False  # 冻结 gamma/beta
            frozen_count += 1
    return frozen_count


def unfreeze_bn_layers(model):
    """解冻所有 BN 层"""
    unfrozen_count = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            module.train()
            for param in module.parameters():
                param.requires_grad = True
            unfrozen_count += 1
    return unfrozen_count


def evaluate_model(fabric, model, cfg, evaluators, eval_pipeline, epoch_label):
    """评估模型并返回结果"""
    fabric.barrier()
    all_result = {}

    for evaluator in evaluators:
        if fabric.local_rank == 0:
            print(f"  评估 {evaluator.name}...")
        result = evaluator.evaluate(eval_pipeline, epoch=0, step=0, n_images_seen=0)
        all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})

    if fabric.local_rank == 0 and 'agedb_30/accuracy' in all_result:
        acc = all_result['agedb_30/accuracy']
        print(f"  {epoch_label}: agedb_30 = {acc:.4f} ({acc*100:.2f}%)")

    fabric.barrier()
    return all_result


def train_to_progress(fabric, cfg, model, classifier, train_pipeline, dataloader,
                      optimizer, lr_scheduler, target_progress, freeze_bn=False):
    """训练到指定进度 (以总步数百分比计算)"""
    model.train()
    if freeze_bn:
        # 强制 BN 层进入 eval 模式
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()

    batch_length = len(dataloader.dataset) // cfg.trainers.total_batch_size
    total_steps = batch_length * cfg.optims.num_epoch
    target_step = int(total_steps * target_progress)

    current_step = 0
    epoch = 0

    pbar = tqdm(total=target_step, desc=f"训练至 {int(target_progress*100)}%",
                disable=fabric.local_rank != 0)

    while current_step < target_step:
        set_epoch(dataloader, epoch)

        for batch_idx, batch in enumerate(dataloader):
            if current_step >= target_step:
                break

            with fabric.autocast():
                loss = train_pipeline(batch)
                fabric.backward(loss)

            fabric.clip_gradients(model, optimizer, max_norm=cfg.optims.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler_step(lr_scheduler, current_step)
            current_step += 1
            pbar.update(1)

        epoch += 1

    pbar.close()
    fabric.barrier()
    return current_step


if __name__ == '__main__':
    # 解析命令行: 只接受 freeze_bn=True/False
    import sys
    freeze_bn_mode = None
    for arg in sys.argv[1:]:
        if arg.startswith('freeze_bn='):
            freeze_bn_mode = arg.split('=')[1].lower() == 'true'

    if freeze_bn_mode is None:
        print("错误: 必须指定 freeze_bn=True 或 freeze_bn=False")
        sys.exit(1)

    # 手动构建配置（避免 Hydra base.yaml 依赖问题）
    # 先加载所有子配置
    trainers = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/trainers/configs/default.yaml"
    )
    trainers.prefix = f"bn_freeze_exp_{'frozen' if freeze_bn_mode else 'unfrozen'}"
    trainers.num_gpu = 8
    trainers.batch_size = 128
    trainers.num_workers = 8
    trainers.precision = 'bf16-mixed'
    trainers.skip_final_eval = True
    trainers.external_eval = False
    trainers.task = 'topofr'

    # 设置 output_dir
    import datetime
    date_str = datetime.datetime.now().strftime('%m-%d')
    trial = 0
    base_dir = f"/data1/dataset_0605/train_output/{trainers.prefix}_{date_str}_{trial:02d}"
    while os.path.exists(base_dir):
        trial += 1
        base_dir = f"/data1/dataset_0605/train_output/{trainers.prefix}_{date_str}_{trial:02d}"
    trainers.output_dir = base_dir

    # Models
    models = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/models/iresnet_insightface/configs/v1_ir101.yaml"
    )
    models.yaml_path = "/iresnet_insightface/configs/v1_ir101.yaml"
    models.start_from = "/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt"
    models.freeze = False

    # Dataset
    dataset = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/dataset/configs/dataset_0605_train_rec.yaml"
    )
    dataset.model_save_dir = "/data1/dataset_0605/train_output"

    # Data augmentations
    data_augs = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/data_augs/configs/gridsample_v2_numpy.yaml"
    )

    # Classifiers
    classifiers = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/classifiers/configs/partial_fc.yaml"
    )

    # Losses
    losses = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/losses/configs/topofr_adaface.yaml"
    )

    # Pipelines
    pipelines = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/pipelines/configs/train_model_cls_topofr.yaml"
    )

    # Evaluations
    evaluations = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/evaluations/configs/agedb30_only.yaml"
    )

    # Optims
    optims = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/optims/configs/step_sgd.yaml"
    )
    optims.lr = 0.0005
    optims.num_epoch = 15
    optims.warmup_epoch = 2
    optims.lr_milestones = [8, 12]
    optims.momentum = 0.9
    optims.weight_decay = 0.0005
    optims.lr_lambda = 0.1
    optims.max_grad_norm = 5.0
    optims.scheduler = 'step'

    # PEFTs
    pefts = omegaconf.OmegaConf.load(
        "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/pefts/configs/full.yaml"
    )

    # Aligners (必需字段)
    aligners = omegaconf.OmegaConf.create({'name': 'none'})

    # 创建 Config 对象
    cfg = Config(
        trainers=trainers,
        optims=optims,
        models=models,
        dataset=dataset,
        data_augs=data_augs,
        losses=losses,
        classifiers=classifiers,
        aligners=aligners,
        pipelines=pipelines,
        evaluations=evaluations,
        pefts=pefts
    )

    torch.set_float32_matmul_precision(cfg.trainers.float32_matmul_precision)
    torch.backends.cudnn.benchmark = True
    random_utils.setup_seed(seed=cfg.trainers.seed, cuda_deterministic=False)

    # Setup Fabric
    csv_logger = CSVLogger(root_dir=cfg.trainers.output_dir, flush_logs_every_n_steps=1)
    ddp_strategy = DDPStrategy(timeout=datetime.timedelta(minutes=120))
    fabric = Fabric(
        precision=cfg.trainers.precision,
        loggers=[csv_logger],
        accelerator="auto",
        strategy=ddp_strategy,
        devices=cfg.trainers.num_gpu
    )
    fabric.seed_everything(cfg.trainers.seed)
    if cfg.trainers.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset,
                                                   fabric=fabric, seed=cfg.trainers.seed)

    cfg.trainers.local_rank = fabric.local_rank
    cfg.trainers.world_size = fabric.world_size
    if torch.cuda.is_available():
        torch.cuda.set_device(fabric.local_rank)

    if fabric.local_rank == 0:
        print("=" * 80)
        print(f"BN 冻结对比实验: {'冻结 BN' if freeze_bn_mode else '不冻结 BN'}")
        print("=" * 80)
        print(f"预训练模型: {cfg.models.start_from}")
        print(f"数据集: dataset_0605 ({cfg.dataset.num_classes} 类)")
        print(f"训练配置: {cfg.trainers.num_gpu}x GPU, batch_size={cfg.trainers.batch_size}")
        print(f"评估进度: 0% (预训练), 10%, 20%, 30%, 40%, 50%")
        print("=" * 80)
        print()

    # 加载模型
    model = get_model(cfg.models, cfg.trainers.task)
    train_transform = model.make_train_transform()
    test_transform = model.make_test_transform()

    # 数据集
    dataset, label_mapping = get_train_dataset(
        cfg.dataset, train_transform, cfg.data_augs, local_rank=cfg.trainers.local_rank
    )
    dataloader = fabric.setup_dataloader_from_dataset(
        dataset=dataset,
        is_train=True,
        batch_size=cfg.trainers.batch_size,
        num_workers=cfg.trainers.num_workers
    )
    cfg.trainers.total_batch_size = cfg.trainers.batch_size * cfg.trainers.world_size
    batch_length = len(dataloader.dataset) // cfg.trainers.total_batch_size
    cfg.trainers.warmup_step = batch_length * cfg.optims.warmup_epoch
    cfg.trainers.total_step = batch_length * cfg.optims.num_epoch

    # Classifier
    margin_loss_fn = get_margin_loss(cfg.losses)
    classifier = get_classifier(
        cfg.classifiers,
        margin_loss_fn=margin_loss_fn,
        model_cfg=cfg.models,
        num_classes=cfg.dataset.num_classes,
        rank=fabric.local_rank,
        world_size=fabric.world_size
    )

    # PEFT
    apply_peft(model, cfg.pefts, cfg.trainers.local_rank)

    # Setup with Fabric
    if classifier is not None and cfg.classifiers.name != 'partial_fc':
        model, classifier = fabric.setup(model, classifier)
    else:
        model = fabric.setup(model)

    # Optimizer
    if classifier is not None and cfg.classifiers.name == 'partial_fc':
        params = list(model.parameters()) + list(classifier.parameters())
    else:
        params = model.parameters()
    optimizer = make_optimizer(cfg.optims, params)
    optimizer = fabric.setup_optimizers(optimizer)

    lr_scheduler = make_scheduler(cfg.optims, optimizer, cfg.trainers)

    # Pipelines
    train_pipeline = pipeline_from_config(
        cfg.pipelines, 'train',
        model=model, classifier=classifier, aligner=None,
        fabric=fabric, cfg=cfg
    )
    eval_pipeline = pipeline_from_config(
        cfg.pipelines, 'eval',
        model=model, classifier=classifier, aligner=None,
        fabric=fabric, cfg=cfg
    )

    # Evaluators
    evaluators = []
    for eval_name, eval_cfg in cfg.evaluations.per_epoch_evaluations.items():
        evaluator = get_evaluator_by_name(
            eval_cfg.evaluation_type,
            eval_name,
            eval_cfg,
            test_transform,
            cfg.evaluations.data_root,
            fabric
        )
        evaluators.append(evaluator)

    # 冻结 BN (如果需要)
    if freeze_bn_mode:
        frozen_count = freeze_bn_layers(model)
        if fabric.local_rank == 0:
            print(f"已冻结 {frozen_count} 个 BN 层")
            print()

    # 实验结果记录
    results = {}

    # 0. 预训练模型评估
    if fabric.local_rank == 0:
        print("[1/6] 评估预训练模型 (0% 训练进度)...")
    model.eval()
    result_0 = evaluate_model(fabric, model, cfg, evaluators, eval_pipeline, "预训练模型")
    results['0%'] = result_0

    # 训练到各个进度并评估
    progress_points = [0.1, 0.2, 0.3, 0.4, 0.5]
    for idx, target_progress in enumerate(progress_points, start=2):
        if fabric.local_rank == 0:
            print(f"\n[{idx}/6] 训练至 {int(target_progress*100)}% 进度...")

        train_to_progress(
            fabric, cfg, model, classifier, train_pipeline, dataloader,
            optimizer, lr_scheduler, target_progress, freeze_bn=freeze_bn_mode
        )

        if fabric.local_rank == 0:
            print(f"[{idx}/6] 评估 {int(target_progress*100)}% 进度...")
        model.eval()
        result = evaluate_model(
            fabric, model, cfg, evaluators, eval_pipeline,
            f"{int(target_progress*100)}% 训练"
        )
        results[f'{int(target_progress*100)}%'] = result
        model.train()
        if freeze_bn_mode:
            for module in model.modules():
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    module.eval()

    # 保存结果
    if fabric.local_rank == 0:
        output_file = os.path.join(
            cfg.trainers.output_dir,
            f"bn_freeze_results_{'frozen' if freeze_bn_mode else 'unfrozen'}.json"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 提取 agedb_30 准确率
        summary = {}
        for progress, result in results.items():
            if 'agedb_30/accuracy' in result:
                summary[progress] = float(result['agedb_30/accuracy'])

        with open(output_file, 'w') as f:
            json.dump({
                'freeze_bn': freeze_bn_mode,
                'results': summary,
                'full_results': {k: {kk: float(vv) if isinstance(vv, (int, float, torch.Tensor)) else str(vv)
                                     for kk, vv in v.items()} for k, v in results.items()}
            }, f, indent=2)

        print("\n" + "=" * 80)
        print("实验完成！")
        print("=" * 80)
        print(f"BN 状态: {'冻结' if freeze_bn_mode else '不冻结'}")
        print("\nagedb_30 准确率变化:")
        for progress, acc in summary.items():
            print(f"  {progress:>4}: {acc*100:.2f}%")
        print(f"\n结果已保存至: {output_file}")
        print("=" * 80)

#!/usr/bin/env python3
"""
真实训练速度测试：加载真实模型和数据，测试不同 batch size 下的实际速度
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import sys
sys.path.append(os.path.join(root))

import torch
import time
from omegaconf import OmegaConf
from models import get_model
from classifiers import get_classifier
from losses import get_margin_loss
from dataset import get_train_dataset
from pipelines import pipeline_from_name
from optims.optims import make_optimizer
from optims.lr_scheduler import make_scheduler
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from fabric.fabric import setup_dataloader_from_dataset
from tqdm import tqdm

def test_training_speed(batch_size_per_gpu, num_steps=100):
    """
    测试真实训练速度

    Args:
        batch_size_per_gpu: 每个 GPU 的 batch size
        num_steps: 测试步数
    """
    print(f"\n{'='*80}")
    print(f"测试配置: per-GPU batch={batch_size_per_gpu}, 8 GPU")
    print(f"总 batch size = {batch_size_per_gpu} × 8 = {batch_size_per_gpu * 8}")
    print(f"测试步数: {num_steps}")
    print(f"总样本数: {num_steps * batch_size_per_gpu * 8}")
    print(f"{'='*80}\n")

    # 构建配置
    cfg = OmegaConf.create({
        'trainers': {
            'num_gpu': 8,
            'batch_size': batch_size_per_gpu,
            'num_workers': 8,
            'precision': 'bf16-mixed',
            'gradient_acc': 1,
            'world_size': 8,
            'total_batch_size': batch_size_per_gpu * 8,
        },
        'models': {
            'name': 'iresnet_insightface',
            'start_from': '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt',
            'freeze': False,
            'color_space': 'RGB',
            'img_size': [112, 112],
        },
        'dataset': {
            'data_root': '/data1/dataset_0605',
            'rec': 'train_rec',
            'color_space': 'RGB',
            'num_classes': 791509,
            'num_image': 37084481,
        },
        'classifiers': {
            'name': 'fc',
            'sample_rate': 0.40,
            'freeze': False,
        },
        'losses': {
            'margin_loss_name': 'topofr',
            'base_loss': 'adaface',
            'm': 0.4,
            'h': 0.333,
            't_alpha': 0.01,
            'topo_weight': 0.1,
            'use_gum': True,
            'temp': 1,
        },
        'optims': {
            'lr': 0.0005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'max_grad_norm': 5.0,
        },
        'data_augs': {
            'name': 'gridsample_v2_numpy',
        },
        'aligners': {
            'name': 'none',
        },
        'pipelines': {
            'name': 'TrainModelClsTopoFRPipeline',
        },
    })

    # 初始化 Fabric
    fabric = Fabric(
        accelerator='cuda',
        devices=8,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision='bf16-mixed',
    )
    fabric.launch()

    # 加载数据集
    dataset = get_train_dataset(cfg)
    dataloader = setup_dataloader_from_dataset(
        dataset=dataset,
        batch_size=batch_size_per_gpu,
        num_workers=cfg.trainers.num_workers,
        world_size=cfg.trainers.world_size,
        rank=fabric.global_rank,
    )

    # 加载模型
    model = get_model(cfg.models)
    model = fabric.setup(model)

    # 加载分类器
    classifier_cfg = cfg.classifiers
    classifier_cfg.num_classes = cfg.dataset.num_classes
    classifier_cfg.embedding_size = model.config.embedding_size
    classifier_cfg.world_size = cfg.trainers.world_size
    classifier = get_classifier(classifier_cfg)

    # 加载损失函数
    loss_cfg = cfg.losses
    loss_cfg.num_classes = cfg.dataset.num_classes
    margin_loss = get_margin_loss(loss_cfg, classifier.partial_fc)
    classifier.partial_fc.margin_loss = margin_loss
    classifier = fabric.setup(classifier)

    # 加载优化器
    params = list(model.parameters())
    if classifier is not None:
        params += list(classifier.parameters())
    optimizer = make_optimizer(params, cfg.optims)
    optimizer = fabric.setup_optimizers(optimizer)

    # 创建训练 pipeline
    from pipelines.train_model_cls_topofr_pipeline import TrainModelClsTopoFRPipeline
    train_pipeline = TrainModelClsTopoFRPipeline(
        model=model,
        classifier=classifier,
        optimizer=optimizer,
        lr_scheduler=None,
    )
    train_pipeline.train()

    # 数据加载器
    dataloader = fabric.setup_dataloaders(dataloader)

    if fabric.global_rank == 0:
        print("开始训练速度测试...")
        print("-" * 80)

    # Warm up (5步)
    if fabric.global_rank == 0:
        print("Warm up (5步)...")

    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        loss = train_pipeline(batch)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    fabric.barrier()

    if fabric.global_rank == 0:
        print("开始正式测试...")

    # 正式测试
    times = []
    data_times = []
    forward_times = []
    backward_times = []

    tic = time.time()

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        data_time = time.time() - tic

        # 前向
        t0 = time.time()
        with fabric.autocast():
            loss = train_pipeline(batch)
        forward_time = time.time() - t0

        # 反向
        t0 = time.time()
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.optims.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        backward_time = time.time() - t0

        step_time = time.time() - tic

        if fabric.global_rank == 0:
            times.append(step_time)
            data_times.append(data_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)

            if step % 10 == 0:
                print(f"Step {step:3d} | Total: {step_time:.4f}s | "
                      f"Data: {data_time:.4f}s | Fwd: {forward_time:.4f}s | Bwd: {backward_time:.4f}s | "
                      f"Loss: {loss.item():.4f}")

        tic = time.time()

    fabric.barrier()

    # 统计结果
    if fabric.global_rank == 0:
        print("\n" + "="*80)
        print(f"测试结果 (per-GPU batch={batch_size_per_gpu}, 总batch={batch_size_per_gpu * 8}):")
        print("="*80)

        avg_total = sum(times) / len(times)
        avg_data = sum(data_times) / len(data_times)
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)

        print(f"平均每步耗时: {avg_total:.4f}秒")
        print(f"  - 数据加载: {avg_data:.4f}秒 ({avg_data/avg_total*100:.1f}%)")
        print(f"  - 前向传播: {avg_forward:.4f}秒 ({avg_forward/avg_total*100:.1f}%)")
        print(f"  - 反向传播: {avg_backward:.4f}秒 ({avg_backward/avg_total*100:.1f}%)")
        print(f"")
        print(f"训练吞吐量: {1.0/avg_total:.2f} it/s")
        print(f"样本处理速度: {batch_size_per_gpu * 8 / avg_total:.0f} samples/s")
        print(f"")

        # 计算一个 epoch 需要多少时间
        total_samples = 37084481
        steps_per_epoch = total_samples // (batch_size_per_gpu * 8)
        epoch_time_hours = steps_per_epoch * avg_total / 3600
        print(f"预估一个 epoch:")
        print(f"  - 总步数: {steps_per_epoch}")
        print(f"  - 耗时: {epoch_time_hours:.2f} 小时")
        print("="*80 + "\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("真实训练速度测试 - TopoFR")
    print("="*80)

    # 测试不同 batch size
    for batch_size in [128, 64, 32, 16]:
        try:
            test_training_speed(batch_size, num_steps=100)
        except Exception as e:
            print(f"batch_size={batch_size} 测试失败: {e}")
            import traceback
            traceback.print_exc()

        # 清理 GPU 内存
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    print("\n测试完成！")

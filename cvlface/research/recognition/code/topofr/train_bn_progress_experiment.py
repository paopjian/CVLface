#!/usr/bin/env python3
"""Measure agedb_30 during the first epoch with and without frozen BN layers.

The script deliberately does not call ``config.init`` (and therefore does not
start Hydra).  Both experiments construct their own model, classifier and
optimizer from the same pretrained checkpoint.
"""

import argparse
import datetime
import gc
import json
import math
import os
from functools import partial

import omegaconf
import pyrootutils
import torch
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import DDPStrategy
from tqdm import tqdm

root = pyrootutils.setup_root(search_from=__file__, indicator=["__root__.txt"],
                              pythonpath=True, dotenv=True)

import config  # noqa: E402
from aligners import get_aligner  # noqa: E402
from classifiers import get_classifier  # noqa: E402
from dataset import get_train_dataset, set_epoch  # noqa: E402
from evaluations import get_evaluator_by_name  # noqa: E402
from fabric.fabric import setup_dataloader_from_dataset  # noqa: E402
from general_utils import random_utils  # noqa: E402
from losses import get_margin_loss  # noqa: E402
from models import get_model  # noqa: E402
from optims.lr_scheduler import make_scheduler, scheduler_step  # noqa: E402
from optims.optims import make_optimizer  # noqa: E402
from pefts import apply_peft  # noqa: E402
from pipelines import pipeline_from_config, pipeline_from_name  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED = os.path.join(
    root, "pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt"
)


def load_yaml(path):
    return omegaconf.OmegaConf.load(path)


def build_config(args, output_dir):
    """Build the dataclass config without Hydra composition."""
    trainers = load_yaml(os.path.join(HERE, "trainers/configs/default.yaml"))
    trainers.task = "topofr"
    trainers.prefix = "bn_progress"
    trainers.num_gpu = args.num_gpu
    trainers.batch_size = args.batch_size
    trainers.num_workers = args.num_workers
    trainers.precision = args.precision
    trainers.using_wandb = False
    trainers.skip_final_eval = True
    trainers.external_eval = False
    trainers.output_dir = output_dir

    models = load_yaml(os.path.join(HERE, "models/iresnet_insightface/configs/v1_ir101.yaml"))
    models.yaml_path = "/iresnet_insightface/configs/v1_ir101.yaml"
    models.start_from = args.pretrained
    models.freeze = False

    dataset = load_yaml(os.path.join(HERE, "dataset/configs/dataset_0605_train_rec.yaml"))
    dataset.data_root = args.data_root
    dataset.model_save_dir = output_dir

    data_augs = load_yaml(os.path.join(HERE, "data_augs/configs/gridsample_v2_numpy.yaml"))
    classifiers = load_yaml(os.path.join(HERE, "classifiers/configs/partial_fc.yaml"))
    losses = load_yaml(os.path.join(HERE, "losses/configs/topofr_adaface.yaml"))
    pipelines = load_yaml(os.path.join(HERE, "pipelines/configs/train_model_cls_topofr.yaml"))
    pipelines.resume = ""
    evaluations = load_yaml(os.path.join(HERE, "evaluations/configs/agedb30_only.yaml"))
    evaluations.data_root = args.data_root
    pefts = load_yaml(os.path.join(HERE, "pefts/configs/full.yaml"))
    aligners = omegaconf.OmegaConf.create({"name": "none", "start_from": "", "freeze": False})

    optims = load_yaml(os.path.join(HERE, "optims/configs/step_sgd.yaml"))
    optims.lr = args.lr
    optims.num_epoch = 1
    optims.warmup_epoch = 0
    optims.lr_milestones = []
    optims.max_grad_norm = 5.0

    return config.Config(trainers=trainers, optims=optims, models=models,
                         dataset=dataset, data_augs=data_augs, losses=losses,
                         classifiers=classifiers, aligners=aligners,
                         pipelines=pipelines, evaluations=evaluations, pefts=pefts)


def configure_bn(model, freeze_parameters):
    count = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.SyncBatchNorm)):
            if freeze_parameters:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False
            count += 1
    return count


def set_bn_mode(model, frozen):
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.SyncBatchNorm)):
            module.train(not frozen)


def build_components(fabric, cfg, dataset, label_mapping, freeze_bn):
    model = get_model(cfg.models, cfg.trainers.task)
    model = model.to(memory_format=torch.channels_last)
    classifier = get_classifier(
        cfg.classifiers, get_margin_loss(cfg.losses), cfg.models,
        cfg.dataset.num_classes, fabric.local_rank, fabric.world_size,
    )
    aligner = get_aligner(cfg.aligners)
    model, classifier = apply_peft(
        cfg.pefts, model=model, classifier=classifier,
        data_cfg=cfg.dataset, label_mapping=label_mapping,
    )
    # DDP must see the final trainable-parameter set at construction time.
    configure_bn(model, freeze_parameters=freeze_bn)

    optimizer = make_optimizer(cfg, model, classifier, aligner)
    # The scheduler needs the exact number of batches in this one-epoch run.
    scheduler = make_scheduler(cfg, optimizer)
    model, optimizer = fabric.setup(model, optimizer)
    if classifier is not None:
        classifier = fabric.setup(classifier) if classifier.apply_ddp else classifier.to(fabric.device)
    aligner = fabric.setup(aligner) if aligner.has_trainable_params() else aligner.to(fabric.device)

    train_pipeline = pipeline_from_config(
        cfg.pipelines, model, classifier, aligner, optimizer, scheduler,
    )
    train_pipeline.integrity_check(dataset)
    eval_pipeline = pipeline_from_name(cfg.pipelines.eval_pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataset.color_space)
    return model, classifier, aligner, optimizer, scheduler, train_pipeline, eval_pipeline


def make_evaluators(fabric, cfg, eval_pipeline):
    evaluators = []
    for name, info in cfg.evaluations.per_epoch_evaluations.items():
        data_path = os.path.join(cfg.evaluations.data_root, info.path)
        evaluator = get_evaluator_by_name(
            eval_type=info.evaluation_type, name=name, eval_data_path=data_path,
            transform=eval_pipeline.make_test_transform(), fabric=fabric,
            batch_size=info.batch_size * 4, num_workers=info.num_workers,
        )
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluators.append(evaluator)
    return evaluators


def evaluate_agedb(fabric, evaluators, eval_pipeline, label):
    fabric.barrier()
    result = {}
    eval_pipeline.eval()
    for evaluator in evaluators:
        current = evaluator.evaluate(eval_pipeline, epoch=0, step=0, n_images_seen=0)
        if fabric.local_rank == 0:
            result.update({f"{evaluator.name}/{key}": value for key, value in current.items()})
    fabric.barrier()
    accuracy = result.get("agedb_30/acc")
    if fabric.local_rank == 0:
        if accuracy is None:
            raise RuntimeError(f"agedb_30 evaluator returned no acc metric: {result}")
        print(f"{label}: agedb_30 = {float(accuracy):.2f}%")
    return float(accuracy) if accuracy is not None else None


def train_and_evaluate(fabric, cfg, model, optimizer, scheduler, train_pipeline,
                       eval_pipeline, evaluators, dataloader, freeze_bn, total_steps,
                       progress_points):
    mode = "frozen" if freeze_bn else "unfrozen"
    progress_points = tuple(progress_points)
    targets = [max(1, min(total_steps, math.ceil(total_steps * p))) for p in progress_points]
    results = {}
    current_step = 0
    train_pipeline.train()
    set_bn_mode(model, freeze_bn)
    set_epoch(dataloader, 0, cfg)
    iterator = iter(dataloader)
    pbar = tqdm(total=targets[-1], desc=f"训练 {mode}", disable=fabric.local_rank != 0)

    for target_step, progress in zip(targets, progress_points):
        while current_step < target_step:
            batch = next(iterator)
            with fabric.autocast():
                loss = train_pipeline(batch)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=cfg.optims.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler_step(scheduler, current_step)
            current_step += 1
            pbar.update(1)
        model.eval()
        results[f"{int(progress * 100)}%"] = evaluate_agedb(
            fabric, evaluators, eval_pipeline, f"{mode} {int(progress * 100)}%"
        )
        if target_step < targets[-1]:
            train_pipeline.train()
            set_bn_mode(model, freeze_bn)
    pbar.close()
    fabric.barrier()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--data-root", default="/data1/dataset_0605")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-gpu", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed", choices=("bf16-mixed", "16-mixed", "32-true"))
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--progress-start", type=int, default=10,
        help="首个评估进度（百分比，包含）",
    )
    parser.add_argument(
        "--progress-end", type=int, default=50,
        help="最后一个评估进度（百分比，包含）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.pretrained):
        raise FileNotFoundError(f"预训练模型不存在: {args.pretrained}")
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"数据根目录不存在: {args.data_root}")

    if not args.output_dir:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.data_root, "train_output", f"bn_progress_{stamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = build_config(args, args.output_dir)
    if not 1 <= args.progress_start <= args.progress_end <= 100:
        raise ValueError("progress-start/end 必须满足 1 <= start <= end <= 100")
    progress_percentages = tuple(range(args.progress_start, args.progress_end + 1))
    progress_points = tuple(percent / 100.0 for percent in progress_percentages)
    torch.set_float32_matmul_precision(cfg.trainers.float32_matmul_precision)
    random_utils.setup_seed(cfg.trainers.seed, cuda_deterministic=False)

    fabric = Fabric(
        precision=cfg.trainers.precision,
        loggers=[CSVLogger(root_dir=args.output_dir, flush_logs_every_n_steps=1)],
        accelerator="auto",
        strategy=DDPStrategy(timeout=datetime.timedelta(minutes=120)),
        devices=cfg.trainers.num_gpu,
    )
    fabric.seed_everything(cfg.trainers.seed)
    if cfg.trainers.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(
        setup_dataloader_from_dataset, fabric=fabric, seed=cfg.trainers.seed,
    )
    cfg.trainers.local_rank = fabric.local_rank
    cfg.trainers.world_size = fabric.world_size
    if torch.cuda.is_available():
        torch.cuda.set_device(fabric.local_rank)

    # Dataset and dataloader are shared; resetting sampler epoch makes both
    # conditions consume the same first-epoch sample order.
    model_probe = get_model(cfg.models, cfg.trainers.task)
    train_transform = model_probe.make_train_transform()
    test_transform = model_probe.make_test_transform()
    dataset, label_mapping = get_train_dataset(
        cfg.dataset, train_transform, cfg.data_augs, local_rank=fabric.local_rank,
    )
    dataloader = fabric.setup_dataloader_from_dataset(
        dataset=dataset, is_train=True, batch_size=cfg.trainers.batch_size,
        num_workers=cfg.trainers.num_workers,
    )
    del model_probe, test_transform
    total_steps = len(dataloader)
    cfg.trainers.total_batch_size = cfg.trainers.batch_size * cfg.trainers.world_size
    cfg.trainers.total_step = total_steps
    cfg.trainers.warmup_step = 0
    if total_steps < 1:
        raise RuntimeError("训练 dataloader 没有 batch，无法执行实验")
    if fabric.local_rank == 0:
        points_text = ", ".join(f"{percent}%" for percent in progress_percentages)
        print(f"总 batch: {total_steps}; 评估节点: {points_text}")

    all_results = {"pretrained": {}, "frozen_bn": {}, "unfrozen_bn": {}}

    # First condition: baseline evaluation then train with frozen BN.
    # model_probe above only supplies transforms; reset RNG before constructing
    # the actual training components so both conditions initialize identically.
    random_utils.setup_seed(cfg.trainers.seed, cuda_deterministic=False)
    model, classifier, aligner, optimizer, scheduler, train_pipe, eval_pipe = build_components(
        fabric, cfg, dataset, label_mapping, freeze_bn=True,
    )
    evaluators = make_evaluators(fabric, cfg, eval_pipe)
    all_results["pretrained"]["agedb_30"] = evaluate_agedb(
        fabric, evaluators, eval_pipe, "预训练模型基线"
    )
    all_results["frozen_bn"] = train_and_evaluate(
        fabric, cfg, model, optimizer, scheduler, train_pipe, eval_pipe,
        evaluators, dataloader, True, total_steps, progress_points,
    )
    del model, classifier, aligner, optimizer, scheduler, train_pipe, eval_pipe, evaluators
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fabric.barrier()

    # Second condition: a fresh construction reloads the same pretrained file.
    # Recreate the loader so persistent workers start with the same augmentation
    # RNG state and the same sampler order as the frozen-BN run.
    del dataloader
    gc.collect()
    dataloader = fabric.setup_dataloader_from_dataset(
        dataset=dataset, is_train=True, batch_size=cfg.trainers.batch_size,
        num_workers=cfg.trainers.num_workers,
    )
    # Reset RNGs so the freshly initialized PartialFC starts identically.
    random_utils.setup_seed(cfg.trainers.seed, cuda_deterministic=False)
    model, classifier, aligner, optimizer, scheduler, train_pipe, eval_pipe = build_components(
        fabric, cfg, dataset, label_mapping, freeze_bn=False,
    )
    evaluators = make_evaluators(fabric, cfg, eval_pipe)
    all_results["unfrozen_bn"] = train_and_evaluate(
        fabric, cfg, model, optimizer, scheduler, train_pipe, eval_pipe,
        evaluators, dataloader, False, total_steps, progress_points,
    )

    if fabric.local_rank == 0:
        result_path = os.path.join(args.output_dir, "bn_progress_results.json")
        payload = {
            "pretrained": args.pretrained,
            "data_root": args.data_root,
            "num_gpu": fabric.world_size,
            "batch_size_per_gpu": args.batch_size,
            "total_steps_one_epoch": total_steps,
            "progress_points": list(progress_percentages),
            "results_agedb_30_percent": all_results,
        }
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"结果已保存: {result_path}")
        print(json.dumps(all_results, indent=2, ensure_ascii=False))
    fabric.barrier()


if __name__ == "__main__":
    main()

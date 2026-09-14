#!/usr/bin/env python3
"""Evaluate TopoFR after each of the first five training batches."""

import argparse
import datetime
import gc
import json
import math
import os
from functools import partial

import pyrootutils
import torch
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import DDPStrategy

root = pyrootutils.setup_root(search_from=__file__, indicator=["__root__.txt"],
                              pythonpath=True, dotenv=True)

from dataset import get_train_dataset, set_epoch  # noqa: E402
from fabric.fabric import setup_dataloader_from_dataset  # noqa: E402
from general_utils import random_utils  # noqa: E402
from optims.lr_scheduler import scheduler_step  # noqa: E402

from train_bn_progress_experiment import (  # noqa: E402
    DEFAULT_PRETRAINED,
    build_components,
    build_config,
    evaluate_agedb,
    make_evaluators,
    set_bn_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--data-root", default="/data1/dataset_0605")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-gpu", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--lr", type=float, default=5e-4)
    return parser.parse_args()


def train_first_batches(fabric, cfg, model, optimizer, scheduler, train_pipe,
                        eval_pipe, evaluators, dataloader, frozen):
    train_pipe.train()
    set_bn_mode(model, frozen)
    set_epoch(dataloader, 0, cfg)
    iterator = iter(dataloader)
    values = {}
    for step in range(1, 6):
        batch = next(iterator)
        with fabric.autocast():
            loss = train_pipe(batch)
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.optims.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler_step(scheduler, step - 1)
        model.eval()
        values[str(step)] = evaluate_agedb(
            fabric, evaluators, eval_pipe,
            f"{'frozen' if frozen else 'unfrozen'} batch {step}",
        )
        train_pipe.train()
        set_bn_mode(model, frozen)
    return values


def main():
    args = parse_args()
    if not os.path.isfile(args.pretrained):
        raise FileNotFoundError(args.pretrained)
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(args.data_root)
    if not args.output_dir:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.data_root, "train_output", f"bn_first_batches_{stamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = build_config(args, args.output_dir)
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

    # Probe only supplies transforms; the training model is built below.
    from models import get_model
    probe = get_model(cfg.models, cfg.trainers.task)
    dataset, label_mapping = get_train_dataset(
        cfg.dataset, probe.make_train_transform(), cfg.data_augs,
        local_rank=fabric.local_rank,
    )
    del probe
    dataloader = fabric.setup_dataloader_from_dataset(
        dataset=dataset, is_train=True, batch_size=cfg.trainers.batch_size,
        num_workers=cfg.trainers.num_workers,
    )
    cfg.trainers.total_step = len(dataloader)
    cfg.trainers.warmup_step = 0
    results = {}

    for name, frozen in (("frozen_bn", True), ("unfrozen_bn", False)):
        random_utils.setup_seed(cfg.trainers.seed, cuda_deterministic=False)
        model, classifier, aligner, optimizer, scheduler, train_pipe, eval_pipe = build_components(
            fabric, cfg, dataset, label_mapping, freeze_bn=frozen,
        )
        evaluators = make_evaluators(fabric, cfg, eval_pipe)
        results[name] = {
            "baseline": evaluate_agedb(fabric, evaluators, eval_pipe, f"{name} baseline"),
            "after_batches": train_first_batches(
                fabric, cfg, model, optimizer, scheduler, train_pipe, eval_pipe,
                evaluators, dataloader, frozen,
            ),
        }
        del model, classifier, aligner, optimizer, scheduler, train_pipe, eval_pipe, evaluators
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fabric.barrier()
        if name == "frozen_bn":
            del dataloader
            gc.collect()
            dataloader = fabric.setup_dataloader_from_dataset(
                dataset=dataset, is_train=True, batch_size=cfg.trainers.batch_size,
                num_workers=cfg.trainers.num_workers,
            )

    if fabric.local_rank == 0:
        path = os.path.join(args.output_dir, "bn_first_batches_results.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "pretrained": args.pretrained,
                "num_gpu": fabric.world_size,
                "batch_size_per_gpu": args.batch_size,
                "results_agedb_30_percent": results,
            }, handle, indent=2, ensure_ascii=False)
        print(f"结果已保存: {path}")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    fabric.barrier()


if __name__ == "__main__":
    main()

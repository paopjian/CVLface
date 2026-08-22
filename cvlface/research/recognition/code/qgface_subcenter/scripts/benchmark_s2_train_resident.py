"""Benchmark evaluation while the S2 training runtime remains resident.

The script resumes a QGFace training checkpoint, runs a small number of real
training batches, and then evaluates in the same process. A real training
batch is run after every evaluator so all ranks continue participating in DDP
collectives.
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
from functools import partial

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf

from aligners import get_aligner
from classifiers import get_classifier
from dataset import get_train_dataset, set_epoch
from evaluations import get_evaluator_by_name
from fabric.fabric import setup_dataloader_from_dataset
from general_utils.dist_utils import verify_ddp_weights_equal
from losses import get_margin_loss
from models import get_model
from optims.lr_scheduler import get_last_lr, make_scheduler, scheduler_step
from optims.optims import make_qgface_optimizers
from pefts import apply_peft
from pipelines import pipeline_from_config, pipeline_from_name

from benchmark_s2_eval_ablation import instrument_evaluator


DEFAULT_CHECKPOINT = (
    "/data1/dataset_0605/train_output/"
    "qgface_subcenter_s2_body36_0605_08-11_1/"
    "checkpoints_every_epoch/epoch:13_step:507010"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--precision", default="")
    parser.add_argument(
        "--variants",
        default="baseline,no_gc,no_empty_cache,no_gc_no_empty_cache",
        help="comma-separated cleanup variants",
    )
    parser.add_argument(
        "--eval-names",
        default="",
        help="optional comma-separated evaluator names; empty means all configured evaluators",
    )
    parser.add_argument("--output-json", default="/tmp/s2_train_resident_ablation.json")
    parser.add_argument("--output-dir", default="/tmp/s2_train_resident_runtime")
    parser.add_argument(
        "--train-batches",
        type=int,
        default=500,
        help="real training batches before evaluation; keeps the training runtime resident",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override config.trainers.compile_model",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override config.trainers.channels_last",
    )
    return parser.parse_args()


class RuntimeAblation:
    """Temporarily disable and time process-wide cleanup calls."""

    VALID = {"baseline", "no_gc", "no_empty_cache", "no_gc_no_empty_cache"}

    def __init__(self):
        self.original_gc = gc.collect
        self.original_empty_cache = torch.cuda.empty_cache
        self.variant = "baseline"
        self.gc_calls = []
        self.empty_cache_calls = []

    def start(self, variant):
        if variant not in self.VALID:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.gc_calls = []
        self.empty_cache_calls = []

        def timed_gc(*args, **kwargs):
            start = time.perf_counter()
            result = 0 if variant in ("no_gc", "no_gc_no_empty_cache") else self.original_gc(*args, **kwargs)
            self.gc_calls.append((time.perf_counter() - start, result))
            return result

        def timed_empty_cache(*args, **kwargs):
            start = time.perf_counter()
            result = None if variant in ("no_empty_cache", "no_gc_no_empty_cache") else self.original_empty_cache(*args, **kwargs)
            self.empty_cache_calls.append(time.perf_counter() - start)
            return result

        gc.collect = timed_gc
        torch.cuda.empty_cache = timed_empty_cache

    def stop(self):
        gc.collect = self.original_gc
        torch.cuda.empty_cache = self.original_empty_cache

    def stats(self):
        gc_times = [value[0] for value in self.gc_calls]
        return {
            "gc_calls": len(gc_times),
            "gc_total_sec": sum(gc_times),
            "gc_max_sec": max(gc_times, default=0.0),
            "empty_cache_calls": len(self.empty_cache_calls),
            "empty_cache_total_sec": sum(self.empty_cache_calls),
            "empty_cache_max_sec": max(self.empty_cache_calls, default=0.0),
        }


def setup_fabric(cfg, args):
    timeout_minutes = getattr(cfg.trainers, "timeout_minutes", 120)
    strategy = DDPStrategy(timeout=datetime.timedelta(minutes=timeout_minutes))
    fabric = Fabric(
        precision=args.precision or cfg.trainers.precision,
        accelerator="auto",
        strategy=strategy,
        devices=args.devices,
        loggers=[],
    )
    if args.devices == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(
        setup_dataloader_from_dataset,
        fabric=fabric,
        seed=cfg.trainers.seed,
    )
    cfg.trainers.local_rank = fabric.local_rank
    cfg.trainers.world_size = fabric.world_size
    cfg.trainers.num_gpu = args.devices
    return fabric


def build_training_runtime(cfg, args, fabric, checkpoint):
    model = get_model(cfg.models, cfg.trainers.task)
    train_transform = model.make_train_transform()
    dataset, label_mapping = get_train_dataset(
        cfg.dataset,
        train_transform,
        cfg.data_augs,
        local_rank=fabric.local_rank,
    )
    dataloader = fabric.setup_dataloader_from_dataset(
        dataset=dataset,
        is_train=True,
        batch_size=cfg.trainers.batch_size,
        num_workers=cfg.trainers.num_workers,
    )

    cfg.trainers.total_batch_size = cfg.trainers.batch_size * fabric.world_size
    batch_length = len(dataloader)
    cfg.trainers.warmup_step = batch_length * cfg.optims.warmup_epoch
    cfg.trainers.classifier_warmup_step = batch_length * getattr(
        cfg.optims, "classifier_lr_warmup_epoch", cfg.optims.warmup_epoch
    )
    cfg.trainers.total_step = batch_length * cfg.optims.num_epoch

    margin_loss_fn = get_margin_loss(cfg.losses)
    classifier = get_classifier(
        cfg.classifiers,
        margin_loss_fn=margin_loss_fn,
        model_cfg=cfg.models,
        num_classes=cfg.dataset.num_classes,
        rank=fabric.local_rank,
        world_size=fabric.world_size,
    )
    aligner = get_aligner(cfg.aligners)
    model, classifier = apply_peft(
        cfg.pefts,
        model=model,
        classifier=classifier,
        data_cfg=cfg.dataset,
        label_mapping=label_mapping,
    )

    if cfg.trainers.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.trainers.compile_model:
        model = torch.compile(model, dynamic=False)

    model_optimizer, classifier_optimizer = make_qgface_optimizers(
        cfg, model, classifier
    )
    model_lr_scheduler = make_scheduler(
        cfg, model_optimizer, base_lr=cfg.optims.lr, warmup_steps=cfg.trainers.warmup_step
    ) if model_optimizer is not None else None
    classifier_lr_scheduler = make_scheduler(
        cfg,
        classifier_optimizer,
        base_lr=getattr(cfg.optims, "classifier_lr", cfg.optims.lr),
        warmup_steps=cfg.trainers.classifier_warmup_step,
    ) if classifier_optimizer is not None else None

    model, model_optimizer = fabric.setup(model, model_optimizer)
    if classifier.apply_ddp:
        classifier = fabric.setup(classifier)
    else:
        classifier = classifier.to(fabric.device)
    if classifier_optimizer is not None:
        classifier_optimizer = fabric.setup_optimizers(classifier_optimizer)
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)
    else:
        aligner = aligner.to(fabric.device)

    verify_ddp_weights_equal(model)
    if getattr(classifier, "apply_ddp", False):
        verify_ddp_weights_equal(classifier)

    train_pipeline = pipeline_from_config(
        cfg.pipelines,
        model,
        classifier,
        aligner,
        model_optimizer=model_optimizer,
        classifier_optimizer=classifier_optimizer,
        model_lr_scheduler=model_lr_scheduler,
        classifier_lr_scheduler=classifier_lr_scheduler,
        fabric=fabric,
    )
    train_pipeline._model_max_grad_norm = cfg.optims.max_grad_norm
    train_pipeline._classifier_max_grad_norm = getattr(
        cfg.optims, "classifier_max_grad_norm", cfg.optims.max_grad_norm
    )
    # pipeline_from_config restores model, classifier, optimizers, schedulers and queue.
    if train_pipeline.start_epoch == 0:
        raise RuntimeError(f"Checkpoint was not restored: {checkpoint}")
    train_pipeline.integrity_check(dataloader.dataset)
    eval_pipeline = pipeline_from_name(cfg.pipelines.eval_pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataloader.dataset.color_space)
    return train_pipeline, eval_pipeline, dataloader, batch_length


def build_evaluators(cfg, fabric, eval_pipeline):
    evaluators = []
    for name, info in cfg.evaluations.per_epoch_evaluations.items():
        eval_path = os.path.join(cfg.evaluations.data_root, info.path)
        evaluator = get_evaluator_by_name(
            eval_type=info.evaluation_type,
            name=name,
            eval_data_path=eval_path,
            transform=eval_pipeline.make_test_transform(),
            fabric=fabric,
            batch_size=info.batch_size * 4,
            num_workers=info.num_workers,
        )
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluator.config = info
        evaluators.append(evaluator)
    return evaluators


def train_one_batch(train_pipeline, batch, fabric, step_state):
    train_pipeline.train()
    with fabric.autocast():
        loss = train_pipeline(batch)
        fabric.backward(loss)

    fabric.clip_gradients(
        train_pipeline.model,
        train_pipeline.model_optimizer,
        max_norm=train_pipeline._model_max_grad_norm,
    )
    train_pipeline.model_optimizer.step()
    train_pipeline.model_optimizer.zero_grad(set_to_none=True)
    if train_pipeline.classifier_optimizer is not None:
        fabric.clip_gradients(
            train_pipeline.classifier,
            train_pipeline.classifier_optimizer,
            max_norm=train_pipeline._classifier_max_grad_norm,
        )
        train_pipeline.classifier_optimizer.step()
        train_pipeline.classifier_optimizer.zero_grad(set_to_none=True)

    if train_pipeline.model_lr_scheduler is not None:
        scheduler_step(train_pipeline.model_lr_scheduler, step_state["step"])
    if train_pipeline.classifier_lr_scheduler is not None:
        scheduler_step(train_pipeline.classifier_lr_scheduler, step_state["step"])
    step_state["step"] += 1
    step_state["n_images_seen"] += step_state["total_batch_size"]
    return float(loss.detach().float().item())


def run_training_warmup(train_pipeline, dataloader, fabric, train_batches, step_state):
    train_batches = max(1, train_batches)
    held_batch = None
    iterator = iter(dataloader)
    start = time.perf_counter()
    for batch_idx in range(train_batches):
        batch = next(iterator)
        loss = train_one_batch(train_pipeline, batch, fabric, step_state)
        held_batch = batch
        if fabric.local_rank == 0 and (batch_idx + 1) % 100 == 0:
            print(f"[RESIDENT] warmup train batch={batch_idx + 1}/{train_batches} loss={loss:.5f}")
    fabric.barrier()
    if held_batch is None:
        raise RuntimeError("No training batch was read")
    if fabric.local_rank == 0:
        print(f"[RESIDENT] warmup complete: {train_batches} batches in {time.perf_counter() - start:.2f}s")
    return held_batch


def evaluate_with_resident_training(
    variants,
    evaluators,
    eval_pipeline,
    train_pipeline,
    held_batch,
    fabric,
    step_state,
    stage_timings,
):
    runtime = RuntimeAblation()
    results = []
    for variant in variants:
        runtime.start(variant)
        stage_timings.clear()
        timings = {}
        for evaluator in evaluators:
            key = f"{variant}/{evaluator.name}"
            start = time.perf_counter()
            if fabric.local_rank == 0:
                print(f"[RESIDENT] evaluating {key}")
            evaluator.evaluate(
                eval_pipeline,
                epoch=train_pipeline.start_epoch,
                step=step_state["step"],
                n_images_seen=step_state["n_images_seen"],
            )
            elapsed = time.perf_counter() - start
            timings[evaluator.name] = {
                **stage_timings.get(evaluator.name, {}),
                "evaluate_sec": elapsed,
            }
            fabric.barrier()

            # Keep the DDP process group active between long evaluations.
            loss = train_one_batch(train_pipeline, held_batch, fabric, step_state)
            fabric.barrier()
            if fabric.local_rank == 0:
                print(f"[RESIDENT] completed {key} eval={elapsed:.2f}s train_loss={loss:.5f}")

        row = {"variant": variant, "timings": timings, **runtime.stats()}
        results.append(row)
        runtime.stop()
    return results


def main():
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    config_path = os.path.join(checkpoint, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.load(config_path)
    os.makedirs(args.output_dir, exist_ok=True)
    output_parent = os.path.dirname(os.path.abspath(args.output_json))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    cfg.trainers.output_dir = args.output_dir
    cfg.trainers.using_wandb = False
    cfg.trainers.skip_final_eval = True
    cfg.pipelines.resume = checkpoint
    if args.compile is not None:
        cfg.trainers.compile_model = args.compile
    if args.channels_last is not None:
        cfg.trainers.channels_last = args.channels_last

    torch.set_float32_matmul_precision(cfg.trainers.float32_matmul_precision)
    torch.backends.cudnn.benchmark = True
    fabric = setup_fabric(cfg, args)
    fabric.seed_everything(cfg.trainers.seed)

    train_pipeline, eval_pipeline, dataloader, batch_length = build_training_runtime(
        cfg, args, fabric, checkpoint
    )
    evaluators = build_evaluators(cfg, fabric, eval_pipeline)
    selected_names = {name for name in args.eval_names.split(",") if name}
    if selected_names:
        evaluators = [e for e in evaluators if e.name in selected_names]
    if not evaluators:
        raise ValueError("No evaluators selected")
    stage_timings = {}
    for evaluator in evaluators:
        # Install stage timers without changing evaluator behavior.
        instrument_evaluator(evaluator, fabric, stage_timings)

    step_state = {
        "step": int(train_pipeline.step),
        "n_images_seen": int(train_pipeline.n_images_seen),
        "total_batch_size": int(cfg.trainers.total_batch_size),
    }
    held_batch = run_training_warmup(
        train_pipeline, dataloader, fabric, args.train_batches, step_state
    )
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = evaluate_with_resident_training(
        variants,
        evaluators,
        eval_pipeline,
        train_pipeline,
        held_batch,
        fabric,
        step_state,
        stage_timings,
    )
    summary = {
        "checkpoint": checkpoint,
        "batch_length": batch_length,
        "train_batches": args.train_batches,
        "train_start_epoch": train_pipeline.start_epoch,
        "final_step": step_state["step"],
        "final_n_images_seen": step_state["n_images_seen"],
        "variants": results,
    }
    if fabric.local_rank == 0:
        print("[RESIDENT_SUMMARY] " + json.dumps(summary, sort_keys=True))
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    fabric.barrier()


if __name__ == "__main__":
    main()

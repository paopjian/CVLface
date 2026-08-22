"""S2 evaluation benchmark with cleanup and feature-collection ablations.

The evaluator construction mirrors ``train_opt.py``: the evaluation batch
size is multiplied by four, the inference pipeline is built after Fabric setup,
and every rank participates in feature extraction.
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
from evaluations import get_evaluator_by_name
from fabric.fabric import setup_dataloader_from_dataset
from general_utils.config_utils import load_config
from models import get_model
from pipelines import pipeline_from_name


DEFAULT_CHECKPOINT = (
    "/data1/dataset_0605/train_output/"
    "qgface_subcenter_s2_body36_0605_08-11_1/"
    "checkpoints_every_epoch/epoch:13_step:507010"
)
DEFAULT_EVAL_CONFIG = os.path.join(
    ROOT,
    "research/recognition/code/qgface_subcenter/evaluations/configs/val_20260605.yaml",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument(
        "--variant",
        choices=("baseline", "no_gc", "no_empty_cache", "no_gc_no_empty_cache"),
        default="baseline",
    )
    parser.add_argument(
        "--phase",
        choices=("full", "extract_only"),
        default="full",
        help="full runs metrics; extract_only measures feature collection only",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compile a freshly loaded checkpoint; disabled by default to exclude compile time",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match train_opt.py's channels_last setting",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def patch_runtime(variant):
    """Patch only this benchmark process; the training/evaluator source is untouched."""
    original_gc = gc.collect
    original_empty_cache = torch.cuda.empty_cache
    gc_calls = []
    empty_cache_calls = []

    def timed_gc(*args, **kwargs):
        start = time.perf_counter()
        if variant in ("no_gc", "no_gc_no_empty_cache"):
            result = 0
        else:
            result = original_gc(*args, **kwargs)
        gc_calls.append((time.perf_counter() - start, result))
        return result

    def timed_empty_cache(*args, **kwargs):
        start = time.perf_counter()
        if variant in ("no_empty_cache", "no_gc_no_empty_cache"):
            result = None
        else:
            result = original_empty_cache(*args, **kwargs)
        empty_cache_calls.append(time.perf_counter() - start)
        return result

    gc.collect = timed_gc
    torch.cuda.empty_cache = timed_empty_cache
    return original_gc, original_empty_cache, gc_calls, empty_cache_calls


def setup_fabric(args):
    strategy = DDPStrategy(timeout=datetime.timedelta(minutes=120))
    fabric = Fabric(
        precision=args.precision,
        accelerator="auto",
        strategy=strategy,
        devices=args.devices,
        loggers=[],
    )
    if args.devices == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(
        setup_dataloader_from_dataset, fabric=fabric, seed=2048
    )
    return fabric


def setup_pipeline(args, fabric):
    checkpoint = os.path.abspath(args.checkpoint)
    model_config = load_config(os.path.join(checkpoint, "model.yaml"))
    model_config.start_from = ""
    model_config.freeze = False
    model = get_model(model_config, task="qgface_subcenter")
    model.load_state_dict_from_path(os.path.join(checkpoint, "model.pt"))

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, dynamic=False)

    aligner_config = load_config(
        os.path.join(
            ROOT,
            "research/recognition/code/run_v1/aligners/configs/none.yaml",
        )
    )
    aligner = get_aligner(aligner_config)
    model = fabric.setup(model)
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)

    eval_pipeline = pipeline_from_name("infer_model_pipeline", model, aligner)
    eval_pipeline.integrity_check(dataset_color_space="RGB")
    return eval_pipeline


def build_evaluators(args, fabric, eval_pipeline):
    eval_config = OmegaConf.load(args.eval_config)
    evaluators = []
    for name, info in eval_config.per_epoch_evaluations.items():
        path = os.path.join(eval_config.data_root, info.path)
        evaluator = get_evaluator_by_name(
            eval_type=info.evaluation_type,
            name=name,
            eval_data_path=path,
            transform=eval_pipeline.make_test_transform(),
            fabric=fabric,
            # This matches train_opt.py exactly.
            batch_size=info.batch_size * 4,
            num_workers=info.num_workers,
        )
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluators.append(evaluator)
    return evaluators


def instrument_evaluator(evaluator, fabric, timings):
    """Add per-stage timings without changing evaluator behavior."""
    original_evaluate = evaluator.evaluate
    original_extract = getattr(evaluator, "extract", None)
    original_compute = getattr(evaluator, "compute_metric", None)

    def timed_evaluate(*args, **kwargs):
        start = time.perf_counter()
        result = original_evaluate(*args, **kwargs)
        timings.setdefault(evaluator.name, {})["evaluate_sec"] = time.perf_counter() - start
        return result

    evaluator.evaluate = timed_evaluate

    if original_extract is not None:
        def timed_extract(pipeline, flip_images=False):
            start = time.perf_counter()
            result = original_extract(pipeline, flip_images=flip_images)
            key = "extract_flip_sec" if flip_images else "extract_original_sec"
            timings.setdefault(evaluator.name, {})[key] = time.perf_counter() - start
            return result

        evaluator.extract = timed_extract

    if original_compute is not None:
        def timed_compute(collection, collection_flip):
            start = time.perf_counter()
            result = original_compute(collection, collection_flip)
            timings.setdefault(evaluator.name, {})["compute_metric_sec"] = (
                time.perf_counter() - start
            )
            return result

        evaluator.compute_metric = timed_compute


def run_extract_only(evaluators, eval_pipeline, fabric, timings):
    total_start = time.perf_counter()
    for evaluator in evaluators:
        start = time.perf_counter()
        collection = evaluator.extract(eval_pipeline)
        collection_flip = evaluator.extract(eval_pipeline, flip_images=True)
        elapsed = time.perf_counter() - start
        timings.setdefault(evaluator.name, {})["extract_only_sec"] = elapsed
        if fabric.local_rank == 0:
            print(f"[ABLATION] {evaluator.name} extract_only={elapsed:.2f}s")
        del collection, collection_flip
        fabric.barrier()
    return time.perf_counter() - total_start


def run_full(evaluators, eval_pipeline, fabric, timings):
    total_start = time.perf_counter()
    for evaluator in evaluators:
        if fabric.local_rank == 0:
            print(f"Evaluating {evaluator.name}")
        evaluator.evaluate(eval_pipeline, epoch=0, step=0, n_images_seen=0)
        if fabric.local_rank == 0:
            row = timings.get(evaluator.name, {})
            print(f"[ABLATION] {evaluator.name} {json.dumps(row, sort_keys=True)}")
    fabric.barrier()
    return time.perf_counter() - total_start


def main():
    args = parse_args()
    if not os.path.isfile(os.path.join(args.checkpoint, "model.pt")):
        raise FileNotFoundError(args.checkpoint)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    fabric = setup_fabric(args)
    if fabric.local_rank == 0:
        print(f"checkpoint={args.checkpoint}")
        print(f"variant={args.variant} phase={args.phase} compile={args.compile}")

    original_gc, original_empty_cache, gc_calls, empty_cache_calls = patch_runtime(
        args.variant
    )
    timings = {}
    try:
        eval_pipeline = setup_pipeline(args, fabric)
        evaluators = build_evaluators(args, fabric, eval_pipeline)
        for evaluator in evaluators:
            instrument_evaluator(evaluator, fabric, timings)
        if args.phase == "extract_only":
            total_sec = run_extract_only(evaluators, eval_pipeline, fabric, timings)
        else:
            total_sec = run_full(evaluators, eval_pipeline, fabric, timings)

        if fabric.local_rank == 0:
            gc_seconds = sum(item[0] for item in gc_calls)
            empty_seconds = sum(empty_cache_calls)
            summary = {
                "checkpoint": args.checkpoint,
                "variant": args.variant,
                "phase": args.phase,
                "total_sec": total_sec,
                "gc_calls": len(gc_calls),
                "gc_total_sec": gc_seconds,
                "gc_max_sec": max((item[0] for item in gc_calls), default=0.0),
                "empty_cache_calls": len(empty_cache_calls),
                "empty_cache_total_sec": empty_seconds,
                "empty_cache_max_sec": max(empty_cache_calls, default=0.0),
                "timings": timings,
            }
            print("[ABLATION_SUMMARY] " + json.dumps(summary, sort_keys=True))
            if args.output_json:
                with open(args.output_json, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2, ensure_ascii=True)
    finally:
        gc.collect = original_gc
        torch.cuda.empty_cache = original_empty_cache


if __name__ == "__main__":
    main()

"""Compare train_opt's in-process evaluation with external_torch_eval.

Run this file through Fabric for the same number of GPUs used by training, for
example::

    fabric run --devices=8 --precision=bf16-mixed benchmark_eval_speed.py \
        --mode internal --num-gpu 8

Use ``--mode external`` for the subprocess implementation. Both modes use the
same checkpoint and evaluation YAML by default.
"""

import argparse
import datetime
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

# The task modules (models, evaluations, pipelines, ...) live beside train_opt.py,
# while this benchmark is nested under scripts/benchmark.
TASK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_ROOT))

import torch
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import DDPStrategy

from aligners import get_aligner
from evaluations import get_evaluator_by_name, run_combined_evaluations
from external_torch_eval import run_external_torch_eval
from general_utils.config_utils import load_config
from models import get_model
from pefts import apply_peft
from pipelines import pipeline_from_name
from fabric.fabric import setup_dataloader_from_dataset


DEFAULT_CHECKPOINT = (
    "/data1/dataset_0605/train_output/"
    "coreface_subcenter_s2_body36_sgd20_0605_08-03_0/"
    "checkpoints_every_epoch/epoch:12"
)
DEFAULT_EVAL_CONFIG = os.path.join(
    TASK_ROOT, "evaluations", "configs", "val_20260605.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("internal", "external"), required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--num-gpu", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/cvlface_eval_speed")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    parser.add_argument(
        "--no-compile-internal",
        action="store_true",
        help="Do not compile the internal model; train_opt normally compiles it.",
    )
    return parser.parse_args()


def make_fabric(args, name):
    logger = CSVLogger(
        root_dir=os.path.join(args.output_dir, "fabric_logs"),
        name=name,
        flush_logs_every_n_steps=1,
    )
    strategy = DDPStrategy(timeout=datetime.timedelta(minutes=args.timeout_minutes))
    fabric = Fabric(
        precision=args.precision,
        accelerator="auto",
        strategy=strategy,
        devices=args.num_gpu,
        loggers=[logger],
    )
    if args.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(
        setup_dataloader_from_dataset, fabric=fabric, seed=2048
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(fabric.local_rank)
    return fabric


def load_eval_config(path):
    config = load_config(path)
    config.yaml_path = os.path.abspath(path)
    return config


def synchronise_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def checkpoint_epoch(path):
    name = os.path.basename(os.path.normpath(path))
    if name.startswith("epoch:"):
        return int(name.split(":", 1)[1].split("_", 1)[0])
    return 0


def build_internal_evaluators(args, fabric, checkpoint_config):
    checkpoint = Path(args.checkpoint)
    model_config = load_config(str(checkpoint / "model.yaml"))
    model_config.start_from = ""
    model_config.freeze = False
    model = get_model(model_config, checkpoint_config.trainers.task)
    model.load_state_dict_from_path(str(checkpoint / "model.pt"))
    model, _ = apply_peft(
        checkpoint_config.pefts,
        model=model,
        classifier=None,
        data_cfg=checkpoint_config.dataset,
        label_mapping=None,
    )

    # train_opt applies channels_last and compiles before Fabric wraps the model.
    model = model.to(memory_format=torch.channels_last)
    if not args.no_compile_internal:
        model = torch.compile(model, dynamic=False)
    model = fabric.setup(model)

    aligner_config = load_config(
        os.path.join(ROOT, "research", "recognition", "code", "run_v1",
                     "aligners", "configs", "none.yaml")
    )
    aligner = get_aligner(aligner_config)
    eval_pipeline = pipeline_from_name(
        checkpoint_config.pipelines.eval_pipeline_name, model, aligner
    )
    eval_pipeline.integrity_check(checkpoint_config.dataset.color_space)

    evaluators = []
    for name, info in checkpoint_config.evaluations.per_epoch_evaluations.items():
        eval_data_path = os.path.join(checkpoint_config.evaluations.data_root, info.path)
        worker_count = args.num_workers if args.num_workers is not None else info.num_workers
        evaluator = get_evaluator_by_name(
            eval_type=info.evaluation_type,
            name=name,
            eval_data_path=eval_data_path,
            transform=eval_pipeline.make_test_transform(),
            fabric=fabric,
            # This *4 is the train_opt.py internal-evaluation behavior.
            batch_size=info.batch_size * 4,
            num_workers=worker_count,
        )
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluator.config = info
        evaluators.append(evaluator)
    return eval_pipeline, evaluators


def run_internal(args, fabric, checkpoint_config, eval_config):
    checkpoint_config.evaluations = eval_config
    eval_pipeline, evaluators = build_internal_evaluators(args, fabric, checkpoint_config)
    all_result = {}
    timing = {}
    epoch = checkpoint_epoch(args.checkpoint)
    synchronise_cuda()
    total_start = time.perf_counter()
    for evaluator in evaluators:
        fabric.print(f"Evaluating {evaluator.name}")
        synchronise_cuda()
        start = time.perf_counter()
        result = evaluator.evaluate(eval_pipeline, epoch=epoch, step=0, n_images_seen=0)
        synchronise_cuda()
        elapsed = time.perf_counter() - start
        timing[evaluator.name] = elapsed
        all_result.update({evaluator.name + "/" + key: value for key, value in result.items()})

    synchronise_cuda()
    timing["internal_eval_loop_total"] = time.perf_counter() - total_start

    combined_config = getattr(eval_config, "combined_evaluations", None)
    if combined_config and fabric.local_rank == 0:
        start = time.perf_counter()
        all_result.update(
            run_combined_evaluations({item.name: item for item in evaluators}, combined_config)
        )
        timing["combined_evaluations"] = time.perf_counter() - start
    fabric.barrier()
    timing["internal_total_with_combined"] = time.perf_counter() - total_start
    return all_result, timing


def run_external(args, fabric, checkpoint_config, eval_config):
    # run_external_torch_eval launches eval_all_torch_single.py in a clean
    # process tree, matching train_opt.py's external branch exactly.
    output_dir = os.path.join(args.output_dir, "external")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_config.evaluations = eval_config
    checkpoint_config.trainers.output_dir = output_dir
    checkpoint_config.trainers.num_gpu = args.num_gpu
    checkpoint_config.trainers.precision = args.precision
    checkpoint_config.trainers.external_eval_timeout_minutes = args.timeout_minutes
    checkpoint_config.trainers.external_eval_fabric_bin = os.environ.get(
        "FABRIC_BIN", checkpoint_config.trainers.external_eval_fabric_bin
    )
    synchronise_cuda()
    start = time.perf_counter()
    all_result = run_external_torch_eval(
        fabric=fabric,
        cfg=checkpoint_config,
        checkpoint_dir=args.checkpoint,
        epoch=checkpoint_epoch(args.checkpoint),
    )
    synchronise_cuda()
    timing = {"external_wrapper_total": time.perf_counter() - start}
    if fabric.local_rank == 0:
        raw_path = os.path.join(
            output_dir,
            "external_eval",
            f"epoch_{checkpoint_epoch(args.checkpoint)}_raw.json",
        )
        with open(raw_path, "r", encoding="utf-8") as handle:
            child_payload = json.load(handle)
        timing["external_eval_loop_total"] = child_payload["total_elapsed_seconds"]
        timing.update(
            {
                f"child/{key}": value
                for key, value in child_payload.get("timing_results", {}).items()
            }
        )
    return all_result, timing


def write_result(args, all_result, timing):
    def json_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        if hasattr(value, "item"):
            return value.item()
        return value

    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "checkpoint": os.path.abspath(args.checkpoint),
        "eval_config": os.path.abspath(args.eval_config),
        "num_gpu": args.num_gpu,
        "precision": args.precision,
        "internal_batch_size_multiplier": 4 if args.mode == "internal" else 1,
        "timing_seconds": {key: float(value) for key, value in timing.items()},
        "all_result": {key: json_scalar(value) for key, value in all_result.items()},
    }
    path = result_dir / f"{args.mode}_timing.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"计时结果已写入: {path}")
    print(json.dumps(payload["timing_seconds"], ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "model.pt").is_file():
        raise FileNotFoundError(f"model.pt not found: {checkpoint / 'model.pt'}")
    if not Path(args.eval_config).is_file():
        raise FileNotFoundError(f"evaluation config not found: {args.eval_config}")
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_config = load_config(str(checkpoint / "config.yaml"))
    eval_config = load_eval_config(args.eval_config)
    checkpoint_config.trainers.num_gpu = args.num_gpu
    checkpoint_config.trainers.precision = args.precision
    fabric = make_fabric(args, f"benchmark_{args.mode}")
    if args.mode == "internal":
        all_result, timing = run_internal(args, fabric, checkpoint_config, eval_config)
    else:
        all_result, timing = run_external(args, fabric, checkpoint_config, eval_config)
    if fabric.local_rank == 0:
        write_result(args, all_result, timing)
    fabric.barrier()


if __name__ == "__main__":
    main()

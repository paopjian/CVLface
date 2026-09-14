# CLAUDE.md

使用conda的cvlface环境运行代码

使用中文进行交互

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `work_0605` experiment workspace within CVLface_folder — a face recognition training and evaluation framework built on PyTorch and Lightning Fabric. It focuses on fine-tuning pretrained face recognition backbones (primarily iResNet-101 from AdaFace/WebFace12M) on custom datasets, with multi-GPU DDP support.

The Python import root is `cvlface/` (located at `/root/zhaokj/CVLface_folder/cvlface/`, marked by `__root__.txt`). `pyrootutils` adds this to `sys.path`, enabling imports like `from general_utils import ...`.

This repo variant (`CVLface_folder`) adds LMDB as a data format alongside the original MXNet RecordIO.

## Commands

### Training (multi-GPU DDP, primary usage pattern)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
    --strategy=ddp --devices=8 --precision="bf16-mixed" \
    train5.py \
    trainers.prefix=my_run \
    trainers.num_gpu=8 trainers.batch_size=256 trainers.num_workers=8 \
    trainers.precision='bf16-mixed' \
    models=iresnet/configs/v1_ir101.yaml \
    dataset=configs/dataset_0213_train.yaml \
    data_augs=configs/gridsample_v1.yaml \
    classifiers=configs/partial_fc_sample10.yaml \
    losses=configs/adaface.yaml \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.42 \
    optims=configs/step_sgd.yaml \
    evaluations=configs/val_20260213.yaml \
    dataset.model_save_dir=/data2/dataset_0213_rec/train_output
```
`trainers.num_gpu` **must** match `--devices`. `train5.py` is the primary training script (adds early stopping, per-epoch checkpoints, gradient accumulation over `train.py`).

### Training (single GPU)
```bash
python train5.py trainers.prefix=my_run trainers.num_gpu=1 trainers.batch_size=256 ...
```

### Quick Debug Run
```bash
python train.py \
    trainers=configs/debug models=vit/configs/v1_base \
    data_augs=configs/gridsample_v1 dataset=configs/casia \
    pipelines=configs/train_model_cls classifiers=configs/partial_fc \
    evaluations=configs/quick trainers.batch_size=8 trainers.limit_num_batch=128
```

### Evaluation (single GPU)
```bash
python eval.py --num_gpu 1 --eval_config_name full --ckpt_dir <path_to_checkpoint>
```

### Evaluation (multi-GPU)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fabric run \
    --strategy=ddp --devices=4 --precision="32-true" \
    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir <path_to_checkpoint>
```

### LMDB Data Bundling
```bash
python ../../../../data_utils/recognition/training_data/bundle_images_into_lmdb.py \
    --source_dir /path/to/label_dirs \
    --save_dir /path/to/save \
    --quality 100 \
    --map_size 1099511627776
```

Example scripts are in `scripts/examples/`.

## Configuration System

**Hydra + OmegaConf**. `base.yaml` defines top-level defaults. Each component has its own `configs/` subdirectory with YAML presets. CLI overrides use Hydra syntax (`key=value`). The `.yaml` extension is optional in CLI overrides.

The `Config` dataclass in `config.py` has 11 groups: `trainers`, `optims`, `models`, `dataset`, `data_augs`, `losses`, `classifiers`, `aligners`, `pipelines`, `evaluations`, `pefts`.

Dataset paths resolve via `${oc.env:DATA_ROOT}` or hardcoded paths in custom configs. Experiments output to `cvlface/research/recognition/experiments/<task>/<prefix>_<date>`.

## Architecture

### Training Flow (`train5.py`)
1. `config.init(root)` — Hydra compose from `base.yaml` + CLI overrides
2. `get_model()` — dispatches by yaml_path substring (`/vit/`, `/iresnet/`, `/swin/`, etc.)
3. `get_train_dataset()` — MXNet RecordIO (train.rec/train.idx) or ImageFolder
4. `get_classifier()` — `partial_fc` (with class sampling) or `fc`
5. `apply_peft()` — LoRA, partial_freeze (e.g. `body.42` = unfreeze from block 42+), full, or freeze
6. Lightning Fabric wraps modules for DDP
7. `TrainModelClsPipeline` drives forward/backward; evaluators run per-epoch
8. Early stopping monitors check improvement after each epoch
9. Per-epoch checkpoints saved to `dataset.model_save_dir/.../checkpoints_every_epoch/`
10. Final evaluation loads best checkpoint and runs comprehensive eval

### train5.py vs train.py
- `EarlyStoppingMonitor` — stops after N epochs without improvement
- `MetricImprovementMonitor` — stops if improvement falls below threshold
- `broadcast_should_stop()` — synchronized early stopping for DDP consistency
- Per-epoch checkpoint saving
- Combined evaluations via `run_combined_evaluations()`
- Separate grad_norm tracking for backbone vs classifier
- Gradient accumulation via `trainers.gradient_acc`

### Module Pattern
Every component follows: `__init__.py` (factory function) + `configs/` (YAML presets) + `base/` (abstract class) + concrete implementations. `BaseModel` provides `save_pretrained()`, `load_state_dict_from_path()`, `has_trainable_params()`.

### Data Formats
- **MXNet RecordIO** (train.rec/train.idx) — current primary training format
- **LMDB** — bundling tool at `cvlface/data_utils/recognition/training_data/bundle_images_into_lmdb.py`
  - Format: `[4 bytes: label int32 LE][N bytes: JPEG image]`
  - Classes: `LMDBWriter` (writes with periodic commits), `LMDBReader` (readonly optimized)
  - Metadata keys: `__len__`, `__num_classes__`, `__label_map__`

### Pipelines
- **Train**: `TrainModelClsPipeline`, `TrainKeypointModelClsPipeline`
- **Inference**: `InferModelPipeline`, `InferAlignerModelPipeline`, `InferAlignerKeypointModelPipeline`
- `BasePipeline.save()` manages checkpoints (latest + best); resume via `pipelines.resume=<ckpt_dir>`

### Supported Backbones
iResNet (IR18/50/101), iResNet-InsightFace, ViT, ViT-KPRPE, ViT-iRPE, Swin, Swin-KPRPE, PartFViT. Model dispatch is substring-based on `yaml_path` in `models/__init__.py`.

### Evaluation Types
- `verification` — LFW, AgeDB-30, CFP-FP, CPLFW, CALFW
- `ijbbc` — IJB-B/C protocol (`eval_ijbc.py`)
- `tinyface` — TinyFace rank-1/5
- `custom_verification` / `ijbc_custom` — custom dataset evaluations

## Key Details

- All models use 112x112 input; output embedding dim typically 512
- `np.bool = np.bool_` monkey-patch for MXNet 1.9.1 compatibility (top of train.py/eval.py)
- `partial_fc` classifier splits FC weights across GPUs (no DDP wrapper); `fc` uses standard DDP
- AdaFace loss tracks `batch_mean`/`batch_std` for quality-adaptive margin
- Gradient clipping via `optims.max_grad_norm`; logging to CSV (always) + WandB (optional)
- Checkpoints: `model.pt`, `classifier.pt`, `config.yaml`, `model.yaml` in checkpoint dirs
- Pretrained models at `cvlface/pretrained_models/recognition/`
- Performance optimizations applied: `torch.compile`, `cudnn.benchmark=True`, `persistent_workers`, `prefetch_factor=3`
- Hardware target: 8x RTX 4090 (24GB, PCIe); see `optimization_plan.md` for tuning details

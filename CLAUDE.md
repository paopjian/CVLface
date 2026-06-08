# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CVLFace is a face recognition research toolkit from MSU CVLab. It supports multi-GPU training/evaluation with PyTorch Lightning Fabric, Hydra-based configuration, and pre-trained models on Hugging Face Hub.

## Key Commands

All training/eval commands run from the code directory (e.g., `cvlface/research/recognition/code/run_v1/`).

### Installation
```bash
pip install -r requirements.txt
```

### Mock Run (verify installation, single GPU)
```bash
python train.py trainers.prefix=test_run \
    trainers.num_gpu=1 trainers.batch_size=32 \
    trainers.limit_num_batch=128 \
    dataset=configs/synthetic.yaml \
    data_augs=configs/basic_v1.yaml \
    models=iresnet/configs/v1_ir50.yaml \
    pipelines=configs/train_model_cls.yaml \
    evaluations=configs/base.yaml \
    classifiers=configs/fc.yaml \
    optims=configs/step_sgd.yaml \
    losses=configs/cosface.yaml
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
    --strategy=ddp --devices=4 --precision="32-true" \
    train.py trainers.prefix=my_run trainers.num_gpu=4 \
    trainers.batch_size=256 \
    dataset=configs/casia.yaml \
    models=iresnet/configs/v1_ir50.yaml \
    losses=configs/adaface.yaml \
    ...
```

### Evaluation
```bash
python eval.py --num_gpu 1 --eval_config_name full --ckpt_dir <path_to_checkpoint>
```

## Architecture

### Root Marker
`cvlface/__root__.txt` — pyrootutils uses this to set the project root and python path. All imports resolve relative to `cvlface/`.

### Directory Structure
```
cvlface/
├── general_utils/         # Shared utilities (config, dist, huggingface, img, os, random)
├── data_utils/            # Data loading for recognition (MXNet RecordIO format)
├── apps/                  # Standalone apps (face_alignment, verification)
└── research/recognition/code/
    ├── run_v1/            # Stable training code (recommended baseline)
    └── 0605/             # Experimental branch
```

### Training Code Layout (`run_v1/`)
Each subdirectory is a modular component with its own `configs/` folder of YAML files:
- `models/` — Architecture definitions (iresnet, vit, vit_kprpe, swin, swin_kprpe, part_fvit, etc.)
- `losses/` — Margin losses (adaface, arcface, cosface)
- `classifiers/` — FC heads (full FC, partial FC with sampling)
- `dataset/` — Dataset specs (casia, webface4m, webface12m, synthetic)
- `optims/` — Optimizer + scheduler (cosine, step_sgd, poly)
- `data_augs/` — Augmentation strategies
- `evaluations/` — Benchmark configs (base, full, quick, skip_eval)
- `pipelines/` — Training/inference pipeline logic
- `pefts/` — Parameter-efficient fine-tuning (LoRA, freeze)
- `aligners/` — Face alignment (retinaface, dfa, none)
- `trainers/` — Global training params (batch size, GPU, wandb, precision)

### Configuration System
- Uses **Hydra** with a `base.yaml` defining defaults for all components
- Override any config via command line: `component=path/to/config.yaml` or `component.field=value`
- Config dataclass defined in `config.py` with fields: trainers, optims, models, dataset, data_augs, losses, classifiers, aligners, pipelines, evaluations, pefts
- Experiment output dir: `cvlface/research/experiments/{folder_name}/{prefix}_{Month}-{Date}_{Trial}`

### Environment Variables
Set in a `.env` file at `cvlface/.env` (see `.env_example`):
- `DATA_ROOT` — Path to training datasets (MXNet RecordIO format)
- `HF_TOKEN` — Hugging Face token for model access
- `WANDB_TOKEN` — Weights & Biases token

### Data Format
Training datasets use MXNet RecordIO format. The dataset config references a subdirectory under `DATA_ROOT` via the `rec` field.

### Key Design Patterns
- `pyrootutils` resolves paths from `__root__.txt` marker — all scripts start with this setup
- numpy compatibility shims (`np.bool = np.bool_`) fix mxnet 1.9.1 deprecation warnings
- `trainers.batch_size` is per-GPU; set `trainers.num_gpu` to match `--devices`
- Models are loaded dynamically via config path (e.g., `models=vit_kprpe/configs/v1_base_kprpe_splithead_unshared.yaml`)
- Copy-folder workflow: duplicate `run_v1` for new experiments, keep original as stable baseline

## Core Dependencies
- PyTorch + Lightning Fabric 2.1.3 (distributed training)
- Hydra 1.3.2 (configuration)
- mxnet 1.9.1 (RecordIO data loading)
- timm 0.9.7 (vision model building blocks)
- wandb (experiment tracking)
- transformers + diffusers (auxiliary models)

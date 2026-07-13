# Repository Guidelines

## Project Structure & Module Organization
Core training code lives at the repository root: `train.py` is the main entrypoint, while `eval.py`, `eval_all.py`, and `eval_all_2*.py` cover evaluation flows. Reusable packages are grouped by role: `models/`, `dataset/`, `data_augs/`, `classifiers/`, `aligners/`, `losses/`, `optims/`, `pipelines/`, `trainers/`, and `evaluations/`. Most modules keep YAML configs beside the code in `*/configs/`. Utility shell entrypoints live in `scripts/examples/`, `scripts/eval/`, and `scripts/debug/`. Treat `wandb/`, notebooks, and ad hoc experiment notes as generated or personal artifacts, not primary source.

## Build, Test, and Development Commands
Use Python entrypoints directly for local iteration:

```bash
python train.py
python train.py trainers.prefix=debug trainers.num_gpu=1 models=vit/configs/v1_base.yaml
python eval.py --num_gpu 1 --eval_config_name full --ckpt_dir /path/to/checkpoint
python test_model.py
python test_chkpoint.py
```

`train.py` composes configs from `base.yaml` and per-module YAML overrides. Multi-GPU runs typically use Lightning Fabric, for example `bash scripts/examples/run_vit_kprpe_webface4m.sh`. Reuse the scripts in `scripts/eval/` when validating pretrained checkpoints.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise module names such as `train_keypoint_model_cls_pipeline.py`. Keep config filenames descriptive and lowercase, for example `models/vit/configs/v1_base.yaml`. There is no enforced formatter or linter in this tree, so match nearby code and keep imports and comments minimal and readable.

## Testing Guidelines
This repo uses script-based validation rather than a formal `pytest` suite. Run `python test_model.py` after model or config changes to confirm tensor shapes and model construction. Use `python test_chkpoint.py` when touching checkpoint serialization or resume logic. For training changes, include at least one small sanity run with reduced batch size or `trainers.limit_num_batch`.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase summaries such as `updating documentations` and `cleaning commit history`. Prefer concise, imperative commit messages focused on one change. Pull requests should list the affected modules and configs, summarize the training or evaluation command used for validation, and attach key metrics or screenshots when the change affects results or logging output.

## Configuration Tips
Prefer config overrides to hard-coded paths. New experiments should add or reuse YAML files under the relevant `*/configs/` directory and keep dataset or checkpoint paths outside committed defaults where possible.

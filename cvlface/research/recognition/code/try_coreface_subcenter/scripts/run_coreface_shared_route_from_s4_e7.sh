#!/usr/bin/env bash
# Resume the existing S4 epoch 7 checkpoint and enable clean-feature shared routing at epoch 8.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPU="${NUM_GPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/val_20260605.yaml}"
PREFIX="${PREFIX:-coreface_subcenter_s4_shared_route_from_e7_0605}"
SOURCE_PREFIX="${SOURCE_PREFIX:-coreface_subcenter_s4_joint_after_s2_sgd20_0605}"
SOURCE_EPOCH="${SOURCE_EPOCH:-7}"
S4_E7_CKPT="${S4_E7_CKPT:-}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-true}"

export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -n "${CVLFACE_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$CVLFACE_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ -d "/root/anaconda3/envs/cvlface/lib" ]]; then
    export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "${FABRIC_BIN:-}" ]]; then
    FABRIC=("$FABRIC_BIN")
elif command -v fabric >/dev/null 2>&1; then
    FABRIC=("$(command -v fabric)")
else
    echo "错误: 找不到 fabric，请先激活 cvlface 环境或设置 FABRIC_BIN" >&2
    exit 1
fi

find_source_checkpoint() {
    local line
    local checkpoint
    while IFS= read -r line; do
        checkpoint="${line#* }"
        if [[ -f "$checkpoint/model.pt" \
              && -f "$checkpoint/pipeline.pt" \
              && -f "$checkpoint/classifier_rank0.pt" ]]; then
            printf '%s\n' "$checkpoint"
            return 0
        fi
    done < <(find "$OUTPUT_ROOT" \
        -mindepth 4 -maxdepth 4 -type f \
        -path "$OUTPUT_ROOT/${SOURCE_PREFIX}_*/checkpoints_every_epoch/epoch:${SOURCE_EPOCH}/pipeline.pt" \
        -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr)
    return 1
}

if [[ -z "$S4_E7_CKPT" ]]; then
    S4_E7_CKPT="$(find_source_checkpoint || true)"
fi
if [[ -z "$S4_E7_CKPT" ]]; then
    echo "错误: 未找到 $SOURCE_PREFIX 的 S4 epoch:$SOURCE_EPOCH checkpoint" >&2
    echo "可通过 S4_E7_CKPT=/path/to/checkpoint 显式指定" >&2
    exit 1
fi
if [[ ! -f "$S4_E7_CKPT/model.pt" \
      || ! -f "$S4_E7_CKPT/pipeline.pt" \
      || ! -f "$S4_E7_CKPT/classifier_rank0.pt" ]]; then
    echo "错误: checkpoint 不完整: $S4_E7_CKPT" >&2
    exit 1
fi

echo "项目目录: $PROJECT_DIR"
echo "恢复 checkpoint: $S4_E7_CKPT"
echo "新实验前缀: $PREFIX"
echo "训练范围: S4 epoch:$((SOURCE_EPOCH + 1))-19"
echo "共享路由: clean feature 选择正类子中心，两个 dropout view 锁定该中心"

"${FABRIC[@]}" run \
    --devices="$NUM_GPU" \
    --precision="$PRECISION" \
    train_opt.py \
    "trainers.prefix=$PREFIX" \
    "trainers.num_gpu=$NUM_GPU" \
    "trainers.batch_size=$BATCH_SIZE" \
    "trainers.num_workers=$NUM_WORKERS" \
    "trainers.precision=$PRECISION" \
    "trainers.resume=$S4_E7_CKPT" \
    "trainers.external_eval=True" \
    "trainers.using_wandb=True" \
    "trainers.early_stopping_enabled=False" \
    "trainers.skip_final_eval=$SKIP_FINAL_EVAL" \
    "models=iresnet/configs/v1_ir101.yaml" \
    "models.start_from=$S4_E7_CKPT/model.pt" \
    "models.freeze=False" \
    "dataset=configs/dataset_0605_train_rec.yaml" \
    "dataset.model_save_dir=$OUTPUT_ROOT" \
    "data_augs=configs/gridsample_v2_numpy.yaml" \
    "classifiers=configs/partial_fc_subcenter_k3_sample40.yaml" \
    "classifiers.freeze=False" \
    "losses=configs/adaface_coreface.yaml" \
    "pipelines=configs/train_model_cls_coreface_shared_route.yaml" \
    "pefts=configs/full.yaml" \
    "pefts.classifier_ckpt_dir=$S4_E7_CKPT" \
    "optims=configs/step_sgd.yaml" \
    "optims.scheduler=cosine" \
    "optims.lr=0.0004" \
    "optims.model_lr=0.0004" \
    "optims.classifier_lr=0.000025" \
    "optims.num_epoch=20" \
    "optims.warmup_epoch=3" \
    "optims.momentum=0.9" \
    "optims.weight_decay=0.0005" \
    "optims.lr_lambda=0.1" \
    "optims.max_grad_norm=5.0" \
    "evaluations=$EVAL_CONFIG" \
    "evaluations.eval_every_n_epochs=1"

echo "共享路由 S4 实验完成。"

#!/usr/bin/env bash
# Retrain stage 4 from the stage 3 checkpoint with jointly trainable backbone and classifier.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPU="${NUM_GPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/val_20260605.yaml}"
STAGE3_CKPT="${STAGE3_CKPT:-}"
PREFIX="${PREFIX:-coreface_s4_2_joint_0605}"

MODEL_LR="${MODEL_LR:-0.0008}"
CLASSIFIER_LR="${CLASSIFIER_LR:-0.00005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-true}"
EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-false}"

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

mkdir -p "$OUTPUT_ROOT"

find_latest_epoch_checkpoint() {
    local prefix="$1"
    local line
    local checkpoint
    while IFS= read -r line; do
        checkpoint="${line#* }"
        if [[ -f "$checkpoint/model.pt" && -f "$checkpoint/pipeline.pt" ]]; then
            printf '%s\n' "$checkpoint"
            return 0
        fi
    done < <(find "$OUTPUT_ROOT" \
        -mindepth 4 -maxdepth 4 -type f \
        -path "$OUTPUT_ROOT/${prefix}_*/checkpoints_every_epoch/epoch:*/pipeline.pt" \
        -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr)
    return 1
}

checkpoint_epoch() {
    local checkpoint_name="${1##*/}"
    local epoch="${checkpoint_name#epoch:}"
    if [[ "$checkpoint_name" != epoch:* || ! "$epoch" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    printf '%s\n' "$epoch"
}

if [[ -z "$STAGE3_CKPT" ]]; then
    STAGE3_CKPT="$(find_latest_epoch_checkpoint coreface_s3_classifier_0605 || true)"
fi
if [[ -z "$STAGE3_CKPT" || ! -f "$STAGE3_CKPT/model.pt" || ! -f "$STAGE3_CKPT/classifier_rank0.pt" ]]; then
    echo "错误: 未找到有效的阶段3 checkpoint；可通过 STAGE3_CKPT 显式指定" >&2
    exit 1
fi

STAGE3_EPOCH="$(checkpoint_epoch "$STAGE3_CKPT")" || {
    echo "错误: 阶段3 checkpoint 目录名不包含有效 epoch: $STAGE3_CKPT" >&2
    exit 1
}
if (( STAGE3_EPOCH < 4 )); then
    echo "错误: 阶段3尚未完成，当前 checkpoint 为 epoch:$STAGE3_EPOCH" >&2
    exit 1
fi

RESUME_ARGS=()
STAGE4_2_CKPT="$(find_latest_epoch_checkpoint "$PREFIX" || true)"
if [[ -n "$STAGE4_2_CKPT" ]]; then
    LAST_EPOCH="$(checkpoint_epoch "$STAGE4_2_CKPT")"
    if (( LAST_EPOCH >= NUM_EPOCHS - 1 )); then
        echo "检测到阶段4_2已结束，无需继续训练。"
        echo "最终 checkpoint: $STAGE4_2_CKPT"
        exit 0
    fi
    RESUME_ARGS=("trainers.resume=$STAGE4_2_CKPT")
    echo "将从阶段4_2 epoch:$((LAST_EPOCH + 1)) 恢复完整训练状态。"
fi

echo "阶段3 checkpoint: $STAGE3_CKPT"
echo "模型学习率: $MODEL_LR"
echo "分类器学习率: $CLASSIFIER_LR"
echo "优化器: SGD，调度器: cosine"
echo "weight decay: $WEIGHT_DECAY"
echo "warmup epochs: $WARMUP_EPOCHS"

"${FABRIC[@]}" run \
    --devices="$NUM_GPU" \
    --precision="$PRECISION" \
    train_opt.py \
    "trainers.prefix=$PREFIX" \
    "trainers.num_gpu=$NUM_GPU" \
    "trainers.batch_size=$BATCH_SIZE" \
    "trainers.num_workers=$NUM_WORKERS" \
    "trainers.precision=$PRECISION" \
    "trainers.external_eval=False" \
    "trainers.using_wandb=True" \
    "trainers.early_stopping_enabled=$EARLY_STOPPING_ENABLED" \
    "trainers.skip_final_eval=$SKIP_FINAL_EVAL" \
    "models=iresnet/configs/v1_ir101.yaml" \
    "models.start_from=$STAGE3_CKPT/model.pt" \
    "models.freeze=False" \
    "dataset=configs/dataset_0605_train_rec.yaml" \
    "dataset.model_save_dir=$OUTPUT_ROOT" \
    "data_augs=configs/basic_v2_numpy.yaml" \
    "classifiers=configs/partial_fc_sample10.yaml" \
    "classifiers.sample_rate=0.40" \
    "classifiers.freeze=False" \
    "losses=configs/adaface_coreface.yaml" \
    "pipelines=configs/train_model_cls_coreface.yaml" \
    "pefts=configs/full.yaml" \
    "pefts.classifier_ckpt_dir=$STAGE3_CKPT" \
    "optims=configs/step_sgd.yaml" \
    "optims.scheduler=cosine" \
    "optims.lr=$MODEL_LR" \
    "optims.model_lr=$MODEL_LR" \
    "optims.classifier_lr=$CLASSIFIER_LR" \
    "optims.num_epoch=$NUM_EPOCHS" \
    "optims.warmup_epoch=$WARMUP_EPOCHS" \
    "optims.momentum=0.9" \
    "optims.weight_decay=$WEIGHT_DECAY" \
    "optims.lr_lambda=0.1" \
    "optims.max_grad_norm=5.0" \
    "evaluations=$EVAL_CONFIG" \
    "evaluations.eval_every_n_epochs=1" \
    "${RESUME_ARGS[@]}"

STAGE4_2_CKPT="$(find_latest_epoch_checkpoint "$PREFIX")"
echo "阶段4_2训练完成。"
echo "最终 checkpoint: $STAGE4_2_CKPT"

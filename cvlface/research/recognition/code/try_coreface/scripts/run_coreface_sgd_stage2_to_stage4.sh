#!/usr/bin/env bash
# Start from the completed CoreFace stage 1 and retrain stages 2-4 with SGD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPU="${NUM_GPU:-8}"
BATCH_SIZE_S2="${BATCH_SIZE_S2:-256}"
BATCH_SIZE_S3="${BATCH_SIZE_S3:-512}"
BATCH_SIZE_S4="${BATCH_SIZE_S4:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/val_20260605.yaml}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-true}"
EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-false}"

STAGE1_PREFIX="${STAGE1_PREFIX:-coreface_s1_classifier_0605}"
STAGE2_PREFIX="${STAGE2_PREFIX:-coreface_s2_body36_sgd20_0605}"
STAGE3_PREFIX="${STAGE3_PREFIX:-coreface_s3_classifier_after_s2_sgd20_0605}"
STAGE4_PREFIX="${STAGE4_PREFIX:-coreface_s4_joint_after_s2_sgd20_0605}"

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

COMMON_ARGS=(
    "trainers.num_gpu=$NUM_GPU"
    "trainers.num_workers=$NUM_WORKERS"
    "trainers.precision=$PRECISION"
    "trainers.external_eval=False"
    "trainers.using_wandb=True"
    "trainers.early_stopping_enabled=$EARLY_STOPPING_ENABLED"
    "models=iresnet/configs/v1_ir101.yaml"
    "dataset=configs/dataset_0605_train_rec.yaml"
    "dataset.model_save_dir=$OUTPUT_ROOT"
    "data_augs=configs/basic_v2_numpy.yaml"
    "classifiers=configs/partial_fc_sample10.yaml"
    "classifiers.sample_rate=0.40"
    "losses=configs/adaface_coreface.yaml"
    "pipelines=configs/train_model_cls_coreface.yaml"
    "evaluations=$EVAL_CONFIG"
)

run_stage() {
    local prefix="$1"
    local batch_size="$2"
    shift 2
    echo
    echo "========== 开始 $prefix =========="
    echo "时间: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "每卡 batch size: $batch_size"
    "${FABRIC[@]}" run \
        --devices="$NUM_GPU" \
        --precision="$PRECISION" \
        train_opt.py \
        "trainers.prefix=$prefix" \
        "trainers.batch_size=$batch_size" \
        "${COMMON_ARGS[@]}" \
        "$@"
    echo "========== 完成 $prefix =========="
}

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

stage_is_finished() {
    local checkpoint="$1"
    local final_epoch="$2"
    local epoch
    epoch="$(checkpoint_epoch "$checkpoint")" || return 1
    (( epoch >= final_epoch ))
}

require_stage_finished() {
    local stage="$1"
    local checkpoint="$2"
    local final_epoch="$3"
    if ! stage_is_finished "$checkpoint" "$final_epoch"; then
        echo "错误: Stage $stage 的最后 checkpoint 未达到 epoch:$final_epoch" >&2
        exit 1
    fi
}

if [[ -z "$STAGE1_CKPT" ]]; then
    STAGE1_CKPT="$(find_latest_epoch_checkpoint "$STAGE1_PREFIX" || true)"
fi
if [[ -z "$STAGE1_CKPT" || ! -f "$STAGE1_CKPT/classifier_rank0.pt" ]]; then
    echo "错误: 未找到有效的 Stage 1 checkpoint；可通过 STAGE1_CKPT 显式指定" >&2
    exit 1
fi
require_stage_finished 1 "$STAGE1_CKPT" 4

STAGE2_CKPT="$(find_latest_epoch_checkpoint "$STAGE2_PREFIX" || true)"
STAGE3_CKPT="$(find_latest_epoch_checkpoint "$STAGE3_PREFIX" || true)"
STAGE4_CKPT="$(find_latest_epoch_checkpoint "$STAGE4_PREFIX" || true)"

START_STAGE=2
RESUME_CKPT=""
if [[ -n "$STAGE4_CKPT" ]]; then
    if stage_is_finished "$STAGE4_CKPT" 14; then
        echo "检测到新 Stage 4 已完成，无需继续训练。"
        echo "最终 checkpoint: $STAGE4_CKPT"
        exit 0
    fi
    START_STAGE=4
    RESUME_CKPT="$STAGE4_CKPT"
elif [[ -n "$STAGE3_CKPT" ]]; then
    if stage_is_finished "$STAGE3_CKPT" 4; then
        START_STAGE=4
    else
        START_STAGE=3
        RESUME_CKPT="$STAGE3_CKPT"
    fi
elif [[ -n "$STAGE2_CKPT" ]]; then
    if stage_is_finished "$STAGE2_CKPT" 19; then
        START_STAGE=3
    else
        START_STAGE=2
        RESUME_CKPT="$STAGE2_CKPT"
    fi
fi

echo "Stage 1 checkpoint: $STAGE1_CKPT"
if [[ -n "$RESUME_CKPT" ]]; then
    LAST_EPOCH="$(checkpoint_epoch "$RESUME_CKPT")"
    echo "将从 Stage $START_STAGE epoch:$((LAST_EPOCH + 1)) 恢复完整训练状态。"
else
    echo "将从 Stage $START_STAGE 开始训练。"
fi

# Stage 2：解冻 body.36 之后的 backbone，冻结分类器，SGD + cosine 训练 20 epoch。
if (( START_STAGE <= 2 )); then
    RESUME_ARGS=()
    if (( START_STAGE == 2 )) && [[ -n "$RESUME_CKPT" ]]; then
        RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
    fi
    run_stage "$STAGE2_PREFIX" "$BATCH_SIZE_S2" \
        "models.start_from=$STAGE1_CKPT/model.pt" \
        "models.freeze=True" \
        "classifiers.freeze=True" \
        "pefts=configs/part_freeze.yaml" \
        "pefts.target_modules=body.36" \
        "pefts.classifier_ckpt_dir=$STAGE1_CKPT" \
        "optims=configs/step_sgd.yaml" \
        "optims.lr=0.008" \
        "optims.num_epoch=20" \
        "optims.warmup_epoch=3" \
        "optims.momentum=0.9" \
        "optims.weight_decay=0.0005" \
        "optims.lr_lambda=0.1" \
        "optims.max_grad_norm=5.0" \
        "optims.scheduler=cosine" \
        "evaluations.eval_every_n_epochs=1" \
        "trainers.skip_final_eval=True" \
        "${RESUME_ARGS[@]}"
    STAGE2_CKPT="$(find_latest_epoch_checkpoint "$STAGE2_PREFIX")"
    require_stage_finished 2 "$STAGE2_CKPT" 19
    echo "Stage 2 checkpoint: $STAGE2_CKPT"
fi

# Stage 3：冻结 Stage 2 backbone，沿用原配置重新训练分类器。
if (( START_STAGE <= 3 )); then
    RESUME_ARGS=()
    if (( START_STAGE == 3 )) && [[ -n "$RESUME_CKPT" ]]; then
        RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
    fi
    run_stage "$STAGE3_PREFIX" "$BATCH_SIZE_S3" \
        "models.start_from=$STAGE2_CKPT/model.pt" \
        "models.freeze=True" \
        "classifiers.freeze=False" \
        "pefts=configs/freeze.yaml" \
        "pefts.classifier_ckpt_dir=$STAGE2_CKPT" \
        "optims=configs/step_sgd.yaml" \
        "optims.lr=0.006" \
        "optims.num_epoch=5" \
        "optims.lr_milestones=[2,4]" \
        "trainers.skip_final_eval=True" \
        "${RESUME_ARGS[@]}"
    STAGE3_CKPT="$(find_latest_epoch_checkpoint "$STAGE3_PREFIX")"
    require_stage_finished 3 "$STAGE3_CKPT" 4
    echo "Stage 3 checkpoint: $STAGE3_CKPT"
fi

# Stage 4：完全复用 Stage 4_2 的 backbone + classifier 联合训练参数。
RESUME_ARGS=()
if (( START_STAGE == 4 )) && [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
fi
run_stage "$STAGE4_PREFIX" "$BATCH_SIZE_S4" \
    "models.start_from=$STAGE3_CKPT/model.pt" \
    "models.freeze=False" \
    "classifiers.freeze=False" \
    "pefts=configs/full.yaml" \
    "pefts.classifier_ckpt_dir=$STAGE3_CKPT" \
    "optims=configs/step_sgd.yaml" \
    "optims.scheduler=cosine" \
    "optims.lr=0.0008" \
    "optims.model_lr=0.0008" \
    "optims.classifier_lr=0.00005" \
    "optims.num_epoch=15" \
    "optims.warmup_epoch=3" \
    "optims.momentum=0.9" \
    "optims.weight_decay=0.0005" \
    "optims.lr_lambda=0.1" \
    "optims.max_grad_norm=5.0" \
    "evaluations.eval_every_n_epochs=1" \
    "trainers.skip_final_eval=$SKIP_FINAL_EVAL" \
    "${RESUME_ARGS[@]}"

STAGE4_CKPT="$(find_latest_epoch_checkpoint "$STAGE4_PREFIX")"
require_stage_finished 4 "$STAGE4_CKPT" 14

echo
echo "========== 新流程完成 =========="
echo "最终 checkpoint: $STAGE4_CKPT"

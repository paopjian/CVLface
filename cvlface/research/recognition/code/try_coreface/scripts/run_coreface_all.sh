#!/usr/bin/env bash
# Run the four-stage CoreFace-AdaFace fine-tuning workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 可通过环境变量覆盖；默认值与 流程_coreface.txt 保持一致。
NUM_GPU="${NUM_GPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BATCH_SIZE_S1="${BATCH_SIZE_S1:-512}"
BATCH_SIZE_S2="${BATCH_SIZE_S2:-$BATCH_SIZE}"
BATCH_SIZE_S3="${BATCH_SIZE_S3:-512}"
BATCH_SIZE_S4="${BATCH_SIZE_S4:-$BATCH_SIZE}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/val_20260605.yaml}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-true}"
# 当前四阶段流程固定训满；后续需要恢复早停时设置为 true。
EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-false}"
CORE_FACE_PIPELINE="pipelines=configs/train_model_cls_coreface.yaml"

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

if [[ ! -f "$PRETRAINED_MODEL" ]]; then
    echo "错误: 预训练模型不存在: $PRETRAINED_MODEL" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
    "trainers.num_gpu=$NUM_GPU"
    "trainers.num_workers=$NUM_WORKERS"
    "trainers.precision=$PRECISION"
    "models=iresnet/configs/v1_ir101.yaml"
    "dataset=configs/dataset_0605_train_rec.yaml"
    "data_augs=configs/basic_v2_numpy.yaml"
    "classifiers=configs/partial_fc_sample10.yaml"
    "classifiers.sample_rate=0.40"
    "losses=configs/adaface_coreface.yaml"
    "$CORE_FACE_PIPELINE"
    "evaluations=$EVAL_CONFIG"
    "dataset.model_save_dir=$OUTPUT_ROOT"
    "trainers.external_eval=False"
    "trainers.using_wandb=True"
    "trainers.early_stopping_enabled=$EARLY_STOPPING_ENABLED"
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

latest_epoch_checkpoint() {
    local prefix="$1"
    local result
    if ! result="$(find_latest_epoch_checkpoint "$prefix")"; then
        echo "错误: 未找到 $prefix 的有效 epoch checkpoint" >&2
        exit 1
    fi
    printf '%s\n' "$result"
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
    local stage_output_dir
    local epoch
    stage_output_dir="$(dirname "$(dirname "$checkpoint")")"
    if [[ "${EARLY_STOPPING_ENABLED,,}" == "true" && -f "$stage_output_dir/EARLY_STOPPED" ]]; then
        return 0
    fi
    epoch="$(checkpoint_epoch "$checkpoint")" || return 1
    (( epoch >= final_epoch ))
}

require_stage_finished() {
    local stage="$1"
    local checkpoint="$2"
    local final_epoch="$3"
    if ! stage_is_finished "$checkpoint" "$final_epoch"; then
        echo "错误: 阶段${stage}训练进程已结束，但最后 checkpoint 未到目标 epoch" >&2
        exit 1
    fi
}

echo "项目目录: $PROJECT_DIR"
echo "GPU 数量: $NUM_GPU"
echo "训练数据: dataset/configs/dataset_0605_train_rec.yaml"
echo "输出目录: $OUTPUT_ROOT"

STAGE1_CKPT="$(find_latest_epoch_checkpoint coreface_s1_classifier_0605 || true)"
STAGE2_CKPT="$(find_latest_epoch_checkpoint coreface_s2_body36_0605 || true)"
STAGE3_CKPT="$(find_latest_epoch_checkpoint coreface_s3_classifier_0605 || true)"
STAGE4_CKPT="$(find_latest_epoch_checkpoint coreface_s4_full_0605 || true)"

START_STAGE=1
RESUME_CKPT=""
if [[ -n "$STAGE4_CKPT" ]]; then
    if stage_is_finished "$STAGE4_CKPT" 14; then
        START_STAGE=5
    else
        START_STAGE=4
        RESUME_CKPT="$STAGE4_CKPT"
    fi
elif [[ -n "$STAGE3_CKPT" ]]; then
    if stage_is_finished "$STAGE3_CKPT" 4; then
        START_STAGE=4
    else
        START_STAGE=3
        RESUME_CKPT="$STAGE3_CKPT"
    fi
elif [[ -n "$STAGE2_CKPT" ]]; then
    if stage_is_finished "$STAGE2_CKPT" 14; then
        START_STAGE=3
    else
        START_STAGE=2
        RESUME_CKPT="$STAGE2_CKPT"
    fi
elif [[ -n "$STAGE1_CKPT" ]]; then
    if stage_is_finished "$STAGE1_CKPT" 4; then
        START_STAGE=2
    else
        START_STAGE=1
        RESUME_CKPT="$STAGE1_CKPT"
    fi
fi

if (( START_STAGE == 5 )); then
    echo "检测到阶段4已结束，无需继续训练。"
    echo "最终 checkpoint: $STAGE4_CKPT"
    exit 0
elif [[ -n "$RESUME_CKPT" ]]; then
    LAST_EPOCH="$(checkpoint_epoch "$RESUME_CKPT")"
    echo "检测到阶段${START_STAGE}的中间 checkpoint: $RESUME_CKPT"
    echo "将恢复完整训练状态，并从 epoch:$((LAST_EPOCH + 1)) 继续。"
elif (( START_STAGE > 1 )); then
    echo "检测到阶段$((START_STAGE - 1))已结束，将从阶段${START_STAGE}开始。"
else
    echo "未检测到已有 checkpoint，将从阶段1开始。"
fi

# 阶段1：冻结 backbone，训练分类器。
if (( START_STAGE <= 1 )); then
    RESUME_ARGS=()
    if [[ -n "$RESUME_CKPT" ]]; then
        RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
    fi
    run_stage coreface_s1_classifier_0605 "$BATCH_SIZE_S1" \
        "models.start_from=$PRETRAINED_MODEL" \
        "models.freeze=True" \
        "pefts=configs/freeze.yaml" \
        "optims=configs/step_sgd.yaml" \
        "optims.lr=0.008" \
        "optims.num_epoch=5" \
        "optims.lr_milestones=[2,4]" \
        trainers.skip_final_eval=True \
        "${RESUME_ARGS[@]}"
    STAGE1_CKPT="$(latest_epoch_checkpoint coreface_s1_classifier_0605)"
    require_stage_finished 1 "$STAGE1_CKPT" 4
    echo "阶段1 checkpoint: $STAGE1_CKPT"
fi

# 阶段2：解冻 body.36 之后的 backbone，分类器冻结；第8个 epoch启用 CoreFace。
if (( START_STAGE <= 2 )); then
    RESUME_ARGS=()
    if (( START_STAGE == 2 )) && [[ -n "$RESUME_CKPT" ]]; then
        RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
    fi
    STAGE2_BASE_CKPT="${STAGE1_CKPT:-$STAGE2_CKPT}"
    run_stage coreface_s2_body36_0605 "$BATCH_SIZE_S2" \
        "models.start_from=$STAGE2_BASE_CKPT/model.pt" \
        "models.freeze=True" \
        "classifiers.freeze=True" \
        "pefts=configs/part_freeze.yaml" \
        "pefts.target_modules=body.36" \
        "pefts.classifier_ckpt_dir=$STAGE2_BASE_CKPT" \
        "optims=configs/cosine.yaml" \
        "optims.lr=0.008" \
        "optims.num_epoch=15" \
        "optims.warmup_epoch=2" \
        "evaluations.eval_every_n_epochs=1" \
        trainers.skip_final_eval=True \
        "${RESUME_ARGS[@]}"
    STAGE2_CKPT="$(latest_epoch_checkpoint coreface_s2_body36_0605)"
    require_stage_finished 2 "$STAGE2_CKPT" 14
    echo "阶段2 checkpoint: $STAGE2_CKPT"
fi

# 阶段3：冻结微调后的 backbone，重新训练分类器。
if (( START_STAGE <= 3 )); then
    RESUME_ARGS=()
    if (( START_STAGE == 3 )) && [[ -n "$RESUME_CKPT" ]]; then
        RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
    fi
    STAGE3_BASE_CKPT="${STAGE2_CKPT:-$STAGE3_CKPT}"
    run_stage coreface_s3_classifier_0605 "$BATCH_SIZE_S3" \
        "models.start_from=$STAGE3_BASE_CKPT/model.pt" \
        "models.freeze=True" \
        "pefts=configs/freeze.yaml" \
        "pefts.classifier_ckpt_dir=$STAGE3_BASE_CKPT" \
        "optims=configs/step_sgd.yaml" \
        "optims.lr=0.006" \
        "optims.num_epoch=5" \
        "optims.lr_milestones=[2,4]" \
        trainers.skip_final_eval=True \
        "${RESUME_ARGS[@]}"
    STAGE3_CKPT="$(latest_epoch_checkpoint coreface_s3_classifier_0605)"
    require_stage_finished 3 "$STAGE3_CKPT" 4
    echo "阶段3 checkpoint: $STAGE3_CKPT"
fi

# 阶段4：全量解冻，分类器冻结；第8个 epoch起使用 CoreFace。
RESUME_ARGS=()
if (( START_STAGE == 4 )) && [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARGS=("trainers.resume=$RESUME_CKPT")
fi
STAGE4_BASE_CKPT="${STAGE3_CKPT:-$STAGE4_CKPT}"
run_stage coreface_s4_full_0605 "$BATCH_SIZE_S4" \
    "models.start_from=$STAGE4_BASE_CKPT/model.pt" \
    "models.freeze=False" \
    "pefts=configs/full.yaml" \
    "pefts.classifier_ckpt_dir=$STAGE4_BASE_CKPT" \
    "optims=configs/cosine.yaml" \
    "optims.lr=0.0008" \
    "optims.num_epoch=15" \
    "optims.warmup_epoch=2" \
    "evaluations.eval_every_n_epochs=1" \
    "trainers.skip_final_eval=$SKIP_FINAL_EVAL" \
    "${RESUME_ARGS[@]}"
STAGE4_CKPT="$(latest_epoch_checkpoint coreface_s4_full_0605)"
require_stage_finished 4 "$STAGE4_CKPT" 14

echo
echo "========== 全流程完成 =========="
echo "最终 checkpoint: $STAGE4_CKPT"
echo "step 请从 $STAGE4_CKPT/pipeline.pt 读取，目录名仅包含 epoch。"

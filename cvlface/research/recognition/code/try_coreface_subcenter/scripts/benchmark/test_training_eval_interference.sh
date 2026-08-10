#!/usr/bin/env bash
# Compare TinyFace speed after real training and after an equivalent empty read.
#
# Usage:
#   TEST_CASE=1 bash scripts/benchmark/test_training_eval_interference.sh
#   TEST_CASE=2 bash scripts/benchmark/test_training_eval_interference.sh
#   TEST_CASE=3 bash scripts/benchmark/test_training_eval_interference.sh
#   TEST_CASE=4 bash scripts/benchmark/test_training_eval_interference.sh
#
# Training phases reuse the input checkpoint and never write a new checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$CODE_DIR"

TEST_CASE="${TEST_CASE:-}"
if [[ -z "$TEST_CASE" ]]; then
    echo "错误: 必须设置 TEST_CASE=1、2、3 或 4" >&2
    exit 2
fi
NUM_GPU="${NUM_GPU:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PRECISION="${PRECISION:-bf16-mixed}"
DATA_ROOT="${DATA_ROOT:-/data1/dataset_0605}"
TRAIN_REC="${TRAIN_REC:-train_rec}"
START_CHECKPOINT="${START_CHECKPOINT:-/data1/dataset_0605/train_output/coreface_subcenter_s4_joint_after_s2_sgd20_0605_08-06_0/checkpoints_every_epoch/epoch:10}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/dataset_0605/train_output/tinyface_training_eval_interference_case${TEST_CASE}}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/test_20260605_tinyface.yaml}"
DATA_AUG_CONFIG="${DATA_AUG_CONFIG:-configs/gridsample_v2_numpy.yaml}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-120}"
EVAL_TIMEOUT_MINUTES="${EVAL_TIMEOUT_MINUTES:-90}"
TINYFACE_THREADS="${TINYFACE_THREADS:-32}"

case "$TEST_CASE" in
    1) DEFAULT_MAIN_PORT=29501 ;;
    2) DEFAULT_MAIN_PORT=29502 ;;
    3) DEFAULT_MAIN_PORT=29503 ;;
    4) DEFAULT_MAIN_PORT=29504 ;;
    *) DEFAULT_MAIN_PORT=29504 ;;
esac
MAIN_PORT="${MAIN_PORT:-$DEFAULT_MAIN_PORT}"

export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TINYFACE_NUM_THREADS="$TINYFACE_THREADS"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$MAIN_PORT"

FABRIC_BIN="${FABRIC_BIN:-/root/anaconda3/envs/cvlface/bin/fabric}"
if [[ -d "/root/anaconda3/envs/cvlface/lib" ]]; then
    export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ ! -x "$FABRIC_BIN" ]]; then
    echo "错误: fabric 不存在或不可执行: $FABRIC_BIN" >&2
    exit 1
fi
if [[ ! -d "$START_CHECKPOINT" || ! -f "$START_CHECKPOINT/pipeline.pt" ]]; then
    echo "错误: START_CHECKPOINT 无效，必须包含 pipeline.pt: $START_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/results" "$OUTPUT_DIR/markers"

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

snapshot() {
    local tag="$1"
    echo "timestamp=$(timestamp)"
    echo "--- resources: $tag ---"
    echo "--- free -h ---"
    free -h || true
    echo "--- vmstat ---"
    vmstat 1 2 || true
    echo "--- nvidia-smi ---"
    nvidia-smi || true
}

checkpoint_epoch() {
    local name="${1##*/}"
    local value="${name#epoch:}"
    [[ "$name" == epoch:* && "$value" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$value"
}

run_train() {
    local input_checkpoint="$1"
    local output_dir="$2"
    local epochs="$3"
    local schedule="$4"
    local tag="$5"
    local input_epoch
    input_epoch="$(checkpoint_epoch "$input_checkpoint")"
    local num_epoch=$((input_epoch + epochs + 1))

    mkdir -p "$output_dir"
    snapshot "${tag}_before"
    echo "[$(timestamp)] TRAIN tag=$tag input=$input_checkpoint epochs=$epochs schedule=$schedule num_epoch=$num_epoch"
    BENCHMARK_LIMIT_SCHEDULE="$schedule" \
    "$FABRIC_BIN" run \
        "--devices=$NUM_GPU" \
        "--precision=$PRECISION" \
        "--main-address=127.0.0.1" \
        "--main-port=$MAIN_PORT" \
        "$SCRIPT_DIR/train_opt_interference_test.py" \
        "trainers.prefix=$(basename "$output_dir")" \
        "trainers.output_dir=$output_dir" \
        "trainers.resume=$input_checkpoint" \
        "trainers.num_gpu=$NUM_GPU" \
        "trainers.num_workers=$NUM_WORKERS" \
        "trainers.batch_size=$BATCH_SIZE" \
        "trainers.precision=$PRECISION" \
        "trainers.limit_num_batch=-1" \
        "trainers.benchmark_eval_checkpoint=$input_checkpoint" \
        "trainers.external_eval=True" \
        "trainers.external_eval_compile=True" \
        "trainers.external_eval_timeout_minutes=$EVAL_TIMEOUT_MINUTES" \
        "trainers.timeout_minutes=$TIMEOUT_MINUTES" \
        "trainers.skip_final_eval=True" \
        "trainers.early_stopping_enabled=False" \
        "trainers.using_wandb=False" \
        "optims.num_epoch=$num_epoch" \
        "models=iresnet/configs/v1_ir101.yaml" \
        "dataset=configs/dataset_0605_train_rec.yaml" \
        "dataset.data_root=$DATA_ROOT" \
        "dataset.rec=$TRAIN_REC" \
        "dataset.model_save_dir=$(dirname "$output_dir")" \
        "data_augs=$DATA_AUG_CONFIG" \
        "classifiers=configs/partial_fc_subcenter_k3_sample40.yaml" \
        "losses=configs/adaface_coreface.yaml" \
        "optims=configs/cosine.yaml" \
        "optims.optimizer=sgd" \
        "optims.lr=0.0004" \
        "optims.weight_decay=0.0005" \
        "optims.warmup_epoch=3" \
        "pipelines=configs/train_model_cls_coreface.yaml" \
        "evaluations=$EVAL_CONFIG" \
        "evaluations.data_root=$DATA_ROOT"

    snapshot "${tag}_after"
    printf '%s\t%s\t%s\n' "$(timestamp)" "$tag" "$input_checkpoint"
}

run_empty_epoch() {
    local checkpoint="$1"
    local tag="$2"
    local eval_dir="$OUTPUT_DIR/results/${tag}"
    snapshot "${tag}_before"
    echo "[$(timestamp)] EMPTY_READ tag=$tag checkpoint=$checkpoint"
    "$FABRIC_BIN" run \
        "--devices=$NUM_GPU" \
        "--precision=$PRECISION" \
        "--main-address=127.0.0.1" \
        "--main-port=$MAIN_PORT" \
        "$SCRIPT_DIR/empty_read_tinyface_epochs.py" \
        --data_root "$DATA_ROOT" \
        --rec "$TRAIN_REC" \
        --data_aug_config "$CODE_DIR/data_augs/$DATA_AUG_CONFIG" \
        --num_gpu "$NUM_GPU" \
        --num_workers "$NUM_WORKERS" \
        --batch_size "$BATCH_SIZE" \
        --epochs 1 \
        --checkpoint "$checkpoint" \
        --eval_config_name "$(basename "$EVAL_CONFIG" .yaml)" \
        --precision "$PRECISION" \
        --timeout_minutes "$EVAL_TIMEOUT_MINUTES" \
        --output_dir "$eval_dir" \
        --fabric_bin "$FABRIC_BIN"
    snapshot "${tag}_after"
}

case_one() {
    run_train "$START_CHECKPOINT" "$OUTPUT_DIR/case1_train" 10 \
        "1000,2000,3000,4000,5000,6000,7000,8000,9000,10000" "case1_train_10epochs"
}

case_two() {
    run_train "$START_CHECKPOINT" "$OUTPUT_DIR/case2_train_1" 1 "$FULL_EPOCH_BATCHES" "case2_train_1"
    run_empty_epoch "$START_CHECKPOINT" "case2_empty_1"
    run_train "$START_CHECKPOINT" "$OUTPUT_DIR/case2_train_2" 1 "$FULL_EPOCH_BATCHES" "case2_train_2"
}

case_three() {
    run_empty_epoch "$START_CHECKPOINT" "case3_empty_1"
    run_train "$START_CHECKPOINT" "$OUTPUT_DIR/case3_train_1" 1 "$FULL_EPOCH_BATCHES" "case3_train_1"
    run_empty_epoch "$START_CHECKPOINT" "case3_empty_2"
}

case_four() {
    local EVAL_CONFIG="configs/val_20260605.yaml"
    run_train "$START_CHECKPOINT" "$OUTPUT_DIR/case4_train" 10 \
        "100,100,100,100,100,100,100,100,100,100" "case4_train_100batch_fullval_10epochs"
}

# A non-positive limit makes the copied test trainer consume the complete DataLoader.
FULL_EPOCH_BATCHES="${FULL_EPOCH_BATCHES:--1}"
case "$TEST_CASE" in
    1) case_one ;;
    2) case_two ;;
    3) case_three ;;
    4) case_four ;;
    all)
        case_one
        case_two
        case_three
        case_four
        ;;
    *)
        echo "用法: TEST_CASE=1|2|3|4|all bash $0" >&2
        exit 2
        ;;
esac

echo "[$(timestamp)] 完成 TEST_CASE=$TEST_CASE，结果目录：$OUTPUT_DIR"

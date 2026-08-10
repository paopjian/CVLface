#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$TASK_DIR"

CHECKPOINT="${CHECKPOINT:-/data1/dataset_0605/train_output/coreface_subcenter_s2_body36_sgd20_0605_08-03_0/checkpoints_every_epoch/epoch:12}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-val_20260605}"
NUM_GPU="${NUM_GPU:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/cvlface/bin/python}"
FABRIC_BIN="${FABRIC_BIN:-/root/anaconda3/envs/cvlface/bin/fabric}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/cvlface_eval_speed/torch_compile}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cvlface_eval_speed_matplotlib}"
RESULT_PATH="$OUTPUT_DIR/eval_raw.json"
LOG_PATH="$OUTPUT_DIR/eval.log"

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
export MPLCONFIGDIR
if [[ -d "/root/anaconda3/envs/cvlface/lib" ]]; then
    export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -f "$CHECKPOINT/model.pt" ]]; then
    echo "错误: checkpoint 不存在: $CHECKPOINT/model.pt" >&2
    exit 1
fi

echo "内置入口: eval_all_torch_single.py --compile"
echo "checkpoint: $CHECKPOINT"
echo "eval config: $EVAL_CONFIG_NAME"
echo "GPUs: $NUM_GPU, precision: $PRECISION"

START_NS="$(date +%s%N)"
set +e
"$FABRIC_BIN" run \
    --devices="$NUM_GPU" \
    --precision="$PRECISION" \
    eval_all_torch_single.py \
    --eval_config_name "$EVAL_CONFIG_NAME" \
    --pipeline_name default \
    --single_ckpt_path "$CHECKPOINT" \
    --name "benchmark_torch_compile" \
    --project_name "benchmark_eval_speed" \
    --num_gpu "$NUM_GPU" \
    --precision "$PRECISION" \
    --timeout_minutes "${TIMEOUT_MINUTES:-120}" \
    --result_path "$RESULT_PATH" \
    --compile \
    --timing 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e
END_NS="$(date +%s%N)"
ELAPSED_SEC="$((END_NS - START_NS))"
ELAPSED_SEC="$(awk "BEGIN {printf \"%.3f\", $ELAPSED_SEC / 1000000000}")"

echo "端到端耗时: ${ELAPSED_SEC}s"
echo "日志: $LOG_PATH"
echo "原始计时结果: $RESULT_PATH"
exit "$STATUS"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$TASK_DIR"

CHECKPOINT="${CHECKPOINT:-/data1/dataset_0605/train_output/coreface_subcenter_s2_body36_sgd20_0605_08-03_0/checkpoints_every_epoch/epoch:12}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-val_20260605}"
NUM_GPU="${NUM_GPU:-8}"
TRT_PRECISION="${TRT_PRECISION:-fp16}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/cvlface/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/cvlface_eval_speed/trt}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cvlface_eval_speed_matplotlib}"
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

echo "TRT 入口: eval_all_trt_single.py"
echo "checkpoint: $CHECKPOINT"
echo "eval config: $EVAL_CONFIG_NAME"
echo "GPUs: $NUM_GPU, TensorRT precision: $TRT_PRECISION"
echo "注意: 此脚本每次会重新构建 TensorRT engine。"

START_NS="$(date +%s%N)"
set +e
"$PYTHON_BIN" eval_all_trt_single.py \
    --num_gpu "$NUM_GPU" \
    --eval_config_name "$EVAL_CONFIG_NAME" \
    --ckpt_path "$CHECKPOINT" \
    --name "benchmark_trt" \
    --precision "$TRT_PRECISION" 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e
END_NS="$(date +%s%N)"
ELAPSED_SEC="$((END_NS - START_NS))"
ELAPSED_SEC="$(awk "BEGIN {printf \"%.3f\", $ELAPSED_SEC / 1000000000}")"

echo "端到端耗时: ${ELAPSED_SEC}s"
echo "日志: $LOG_PATH"
echo "TRT 结果目录: $TASK_DIR/eval3_results/benchmark_trt"
exit "$STATUS"

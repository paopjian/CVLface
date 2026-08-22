#!/bin/bash
# Bundle all formats for try3, then run benchmark.
# Usage: bash scripts/benchmark/run_try3.sh
set -e

SOURCE_DIR="/data1/dataset_0605/try3"
SAVE_ROOT="/data1/dataset_0605/benchmark3"
PYTHON="/root/miniconda3/envs/cvlface/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Step 1: Bundle all formats ==="
echo "Source: $SOURCE_DIR"
echo "Save:   $SAVE_ROOT"
echo ""

$PYTHON "$SCRIPT_DIR/bundle_all_fast.py" \
    --source_dir "$SOURCE_DIR" \
    --save_root "$SAVE_ROOT" \
    --formats lmdb,recordio,webdataset,hdf5

echo ""
echo "=== Step 2: Run benchmark ==="
echo ""

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 $PYTHON "$BENCH_DIR/benchmark_dataformat.py" \
    --data_root "$SOURCE_DIR" \
    --benchmark_root "$SAVE_ROOT" \
    --batch_size 256 \
    --num_workers 8 \
    --model ir101

echo ""
echo "=== Done ==="

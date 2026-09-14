#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="turbojpeg"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source "/root/anaconda3/bin/activate" cvlface
cd "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt}"
DATA_ROOT="${DATA_ROOT:-/data1/dataset_0605}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/train_output/bn_progress_1_10_$(date +%Y%m%d_%H%M%S)}"

echo "TopoFR BN progress experiment: 1%-10%"
echo "8 GPUs; one epoch; agedb_30 at every 1%"
echo "pretrained: ${PRETRAINED_MODEL}"
echo "data root:  ${DATA_ROOT}"
echo "output:     ${OUTPUT_DIR}"

fabric run --devices=8 --precision="bf16-mixed" \
  train_bn_progress_experiment.py \
  --num-gpu 8 \
  --pretrained "${PRETRAINED_MODEL}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --progress-start 1 \
  --progress-end 10

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
FABRIC_BIN="${FABRIC_BIN:-/root/anaconda3/envs/cvlface/bin/fabric}"
CONDA_LIB="${CONDA_LIB:-/root/anaconda3/envs/cvlface/lib}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
RESUME_DIR="${RESUME_DIR:-${OUTPUT_ROOT}/subcenter_s5_cls10_0605_07-23_0/checkpoints_every_epoch/epoch:1}"
S4_CKPT="${S4_CKPT:-${OUTPUT_ROOT}/subcenter_s4_joint04_0605_07-22_1/checkpoints_every_epoch/epoch:14}"

[[ -x "${FABRIC_BIN}" ]] || { echo "fabric executable not found: ${FABRIC_BIN}" >&2; exit 2; }
[[ -d "${RESUME_DIR}" ]] || { echo "resume checkpoint not found: ${RESUME_DIR}" >&2; exit 2; }
[[ -d "${S4_CKPT}" ]] || { echo "stage-4 checkpoint not found: ${S4_CKPT}" >&2; exit 2; }
[[ -f "${RESUME_DIR}/pipeline.pt" ]] || { echo "pipeline checkpoint missing: ${RESUME_DIR}/pipeline.pt" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"

"${FABRIC_BIN}" run \
    "--devices=${NUM_GPUS}" \
    --precision=bf16-mixed \
    train_opt.py \
    trainers.prefix=subcenter_s5_cls10_0605_e6 \
    "trainers.num_gpu=${NUM_GPUS}" \
    trainers.num_workers=8 \
    trainers.batch_size=256 \
    trainers.precision=bf16-mixed \
    trainers.skip_final_eval=True \
    "trainers.resume=${RESUME_DIR}" \
    models=iresnet/configs/v1_ir101.yaml \
    "models.start_from=${S4_CKPT}/model.pt" \
    models.freeze=True \
    dataset=configs/dataset_0605_train_rec.yaml \
    "dataset.model_save_dir=${OUTPUT_ROOT}" \
    classifiers=configs/partial_fc_subcenter_k3_sample40.yaml \
    classifiers.sample_rate=1.0 \
    classifiers.freeze=False \
    losses=configs/adaface.yaml \
    data_augs=configs/basic_v2_numpy.yaml \
    pefts=configs/freeze.yaml \
    "pefts.classifier_ckpt_dir=${S4_CKPT}" \
    optims=configs/step_sgd.yaml \
    optims.lr=0.003 \
    optims.num_epoch=6 \
    optims.warmup_epoch=0 \
    "optims.lr_milestones=[1]" \
    optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 \
    evaluations=configs/val_20260605.yaml

echo "stage-5 continuation completed; output root: ${OUTPUT_ROOT}"

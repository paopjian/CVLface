#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
FABRIC_BIN="${FABRIC_BIN:-/root/anaconda3/envs/cvlface/bin/fabric}"
CONDA_LIB="${CONDA_LIB:-/root/anaconda3/envs/cvlface/lib}"
S3_CKPT="${S3_CKPT:-/data1/dataset_0605/train_output/subcenter_s3_cls04_0605_07-21_0/checkpoints_every_epoch/epoch:4}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-/tmp/cvlface_external_eval_smoke}"

if [[ ! -x "${FABRIC_BIN}" ]]; then
    echo "fabric executable not found: ${FABRIC_BIN}" >&2
    exit 2
fi
if [[ ! -f "${S3_CKPT}/model.pt" || ! -f "${S3_CKPT}/classifier_rank0.pt" ]]; then
    echo "stage 3 checkpoint is incomplete: ${S3_CKPT}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cvlface_external_eval_matplotlib}"

"${FABRIC_BIN}" run \
    "--devices=${NUM_GPUS}" \
    --precision=bf16-mixed \
    train_opt.py \
    trainers.prefix=external_torch_eval_smoke_0605 \
    "trainers.num_gpu=${NUM_GPUS}" \
    trainers.batch_size=256 \
    trainers.num_workers=2 \
    trainers.precision=bf16-mixed \
    trainers.limit_num_batch=2 \
    trainers.skip_final_eval=True \
    trainers.using_wandb=True \
    trainers.external_eval=True \
    trainers.external_eval_timeout_minutes=120 \
    models=iresnet/configs/v1_ir101.yaml \
    "models.start_from=${S3_CKPT}/model.pt" \
    models.freeze=False \
    dataset=configs/dataset_0605_train_rec.yaml \
    "dataset.model_save_dir=${SMOKE_OUTPUT_ROOT}" \
    classifiers=configs/partial_fc_subcenter_k3_sample40.yaml \
    classifiers.freeze=False \
    classifiers.sample_rate=0.4 \
    losses=configs/adaface.yaml \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/full.yaml \
    "pefts.classifier_ckpt_dir=${S3_CKPT}" \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0008 \
    optims.num_epoch=1 \
    optims.warmup_epoch=0 \
    optims.momentum=0.9 \
    optims.max_grad_norm=5.0 \
    optims.weight_decay=0.0005 \
    optims.scheduler=cosine \
    evaluations=configs/val_20260605.yaml

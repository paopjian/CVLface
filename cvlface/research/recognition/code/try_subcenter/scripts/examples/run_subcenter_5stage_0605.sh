#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
START_STAGE="${START_STAGE:-1}"
FABRIC_BIN="${FABRIC_BIN:-/root/anaconda3/envs/cvlface/bin/fabric}"
CONDA_LIB="${CONDA_LIB:-/root/anaconda3/envs/cvlface/lib}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt}"

S1_PREFIX="subcenter_s1_cls04_0605"
S2_PREFIX="subcenter_s2_body36_0605"
S3_PREFIX="subcenter_s3_cls04_0605"
S4_PREFIX="subcenter_s4_joint04_0605"
S5_PREFIX="subcenter_s5_cls10_0605"

if [[ ! "${START_STAGE}" =~ ^[1-5]$ ]]; then
    echo "START_STAGE must be an integer from 1 to 5, got: ${START_STAGE}" >&2
    exit 2
fi
if [[ ! -x "${FABRIC_BIN}" ]]; then
    echo "fabric executable not found: ${FABRIC_BIN}" >&2
    exit 2
fi
if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
    echo "pretrained model not found: ${PRETRAINED_MODEL}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"

COMMON_ARGS=(
    "trainers.num_gpu=${NUM_GPUS}"
    "trainers.num_workers=8"
    "trainers.precision=bf16-mixed"
    "models=iresnet/configs/v1_ir101.yaml"
    "dataset=configs/dataset_0605_train_rec.yaml"
    "dataset.model_save_dir=${OUTPUT_ROOT}"
    "classifiers=configs/partial_fc_subcenter_k3_sample40.yaml"
    "losses=configs/adaface.yaml"
    "optims=configs/step_sgd.yaml"
    "optims.momentum=0.9"
    "optims.max_grad_norm=5.0"
    "evaluations=configs/val_20260605.yaml"
)

run_train() {
    "${FABRIC_BIN}" run \
        "--devices=${NUM_GPUS}" \
        --precision=bf16-mixed \
        train_opt.py \
        "${COMMON_ARGS[@]}" \
        "$@"
}

latest_checkpoint() {
    local prefix="$1"
    local run_dir
    local checkpoint_dir

    run_dir="$(
        find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" \
            -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d ' ' -f 2-
    )"
    if [[ -z "${run_dir}" ]]; then
        echo "no completed run found for prefix ${prefix} under ${OUTPUT_ROOT}" >&2
        return 1
    fi

    checkpoint_dir="$(
        find "${run_dir}/checkpoints_every_epoch" -mindepth 1 -maxdepth 1 \
            -type d -name 'epoch:*' -print | sort -V | tail -n 1
    )"
    if [[ -z "${checkpoint_dir}" ]]; then
        echo "no epoch checkpoint found under ${run_dir}/checkpoints_every_epoch" >&2
        return 1
    fi
    if [[ ! -f "${checkpoint_dir}/model.pt" || ! -f "${checkpoint_dir}/classifier_rank0.pt" ]]; then
        echo "incomplete checkpoint: ${checkpoint_dir}" >&2
        return 1
    fi

    printf '%s\n' "${checkpoint_dir}"
}

if (( START_STAGE <= 1 )); then
    run_train \
        "trainers.prefix=${S1_PREFIX}" \
        trainers.batch_size=512 \
        trainers.skip_final_eval=True \
        "models.start_from=${PRETRAINED_MODEL}" \
        models.freeze=True \
        classifiers.freeze=False \
        classifiers.sample_rate=0.4 \
        data_augs=configs/basic_v2_numpy.yaml \
        pefts=configs/freeze.yaml \
        optims.lr=0.008 \
        optims.num_epoch=5 \
        "optims.lr_milestones=[2,4]" \
        optims.weight_decay=0.0001 \
        optims.lr_lambda=0.3
fi
S1_CKPT="$(latest_checkpoint "${S1_PREFIX}")"
echo "stage 1 checkpoint: ${S1_CKPT}"

if (( START_STAGE <= 2 )); then
    run_train \
        "trainers.prefix=${S2_PREFIX}" \
        trainers.batch_size=256 \
        trainers.skip_final_eval=True \
        "models.start_from=${S1_CKPT}/model.pt" \
        models.freeze=True \
        classifiers.freeze=True \
        classifiers.sample_rate=0.4 \
        data_augs=configs/gridsample_v2_numpy.yaml \
        pefts=configs/part_freeze.yaml \
        pefts.target_modules=body.36 \
        "pefts.classifier_ckpt_dir=${S1_CKPT}" \
        optims.lr=0.008 \
        optims.num_epoch=15 \
        optims.warmup_epoch=2 \
        optims.weight_decay=0.0005 \
        optims.lr_lambda=0.1 \
        optims.scheduler=cosine
fi
S2_CKPT="$(latest_checkpoint "${S2_PREFIX}")"
echo "stage 2 checkpoint: ${S2_CKPT}"

if (( START_STAGE <= 3 )); then
    run_train \
        "trainers.prefix=${S3_PREFIX}" \
        trainers.batch_size=512 \
        trainers.skip_final_eval=True \
        "models.start_from=${S2_CKPT}/model.pt" \
        models.freeze=True \
        classifiers.freeze=False \
        classifiers.sample_rate=0.4 \
        data_augs=configs/gridsample_v2_numpy.yaml \
        pefts=configs/freeze.yaml \
        "pefts.classifier_ckpt_dir=${S2_CKPT}" \
        optims.lr=0.006 \
        optims.num_epoch=5 \
        "optims.lr_milestones=[2,4]" \
        optims.weight_decay=0.0001 \
        optims.lr_lambda=0.3
fi
S3_CKPT="$(latest_checkpoint "${S3_PREFIX}")"
echo "stage 3 checkpoint: ${S3_CKPT}"

if (( START_STAGE <= 4 )); then
    run_train \
        "trainers.prefix=${S4_PREFIX}" \
        trainers.batch_size=128 \
        trainers.skip_final_eval=True \
        trainers.external_eval=True \
        "models.start_from=${S3_CKPT}/model.pt" \
        models.freeze=False \
        classifiers.freeze=False \
        classifiers.sample_rate=0.4 \
        data_augs=configs/gridsample_v2_numpy.yaml \
        pefts=configs/full.yaml \
        "pefts.classifier_ckpt_dir=${S3_CKPT}" \
        optims.lr=0.0004 \
        optims.num_epoch=15 \
        optims.warmup_epoch=2 \
        optims.weight_decay=0.0005 \
        optims.lr_lambda=0.1 \
        optims.scheduler=cosine
fi
S4_CKPT="$(latest_checkpoint "${S4_PREFIX}")"
echo "stage 4 checkpoint: ${S4_CKPT}"

if (( START_STAGE <= 5 )); then
    run_train \
        "trainers.prefix=${S5_PREFIX}" \
        trainers.batch_size=256 \
        trainers.skip_final_eval=True \
        "models.start_from=${S4_CKPT}/model.pt" \
        models.freeze=True \
        classifiers.freeze=False \
        classifiers.sample_rate=1.0 \
        data_augs=configs/basic_v2_numpy.yaml \
        pefts=configs/freeze.yaml \
        "pefts.classifier_ckpt_dir=${S4_CKPT}" \
        optims.lr=0.003 \
        optims.num_epoch=2 \
        optims.warmup_epoch=0 \
        "optims.lr_milestones=[1]" \
        optims.weight_decay=0.0001 \
        optims.lr_lambda=0.3
fi
S5_CKPT="$(latest_checkpoint "${S5_PREFIX}")"

echo "five-stage training completed"
echo "final model checkpoint: ${S5_CKPT}/model.pt"
echo "classifier for dataset cleaning: ${S5_CKPT}"

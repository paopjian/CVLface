#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
FABRIC_BIN="${FABRIC_BIN:-/root/anaconda3/envs/cvlface/bin/fabric}"
CONDA_LIB="${CONDA_LIB:-/root/anaconda3/envs/cvlface/lib}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/dataset_0605/train_output}"
S1_CKPT="${S1_CKPT:-/data1/dataset_0605/train_output/subcenter_s1_cls04_0605_07-21_0/checkpoints_every_epoch/epoch:4}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-128}"
S3_BATCH_SIZE="${S3_BATCH_SIZE:-128}"
S4_BATCH_SIZE="${S4_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
EVALUATIONS_CONFIG="${EVALUATIONS_CONFIG:-configs/val_20260605.yaml}"

if [[ ! -x "${FABRIC_BIN}" ]]; then
    echo "fabric executable not found: ${FABRIC_BIN}" >&2
    exit 2
fi
if [[ ! -f "${S1_CKPT}/model.pt" || \
      ! -f "${S1_CKPT}/classifier_rank0.pt" ]]; then
    echo "incomplete stage 1 checkpoint: ${S1_CKPT}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"
export DATA_ROOT="${DATA_ROOT:-/data1/dataset_0605}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
    "trainers.task=qgface_subcenter"
    "trainers.num_gpu=${NUM_GPUS}"
    "trainers.num_workers=${NUM_WORKERS}"
    "trainers.precision=bf16-mixed"
    "trainers.compile_model=${COMPILE_MODEL}"
    "trainers.channels_last=true"
    "trainers.skip_final_eval=true"
    "dataset=configs/dataset_0605_train_rec.yaml"
    "dataset.model_save_dir=${OUTPUT_ROOT}"
    "data_augs=configs/qgface.yaml"
    "losses=configs/qgface_adaface.yaml"
    "pipelines=configs/train_qgface.yaml"
    "evaluations=${EVALUATIONS_CONFIG}"
)

if [[ -n "${LIMIT_NUM_BATCH:-}" ]]; then
    COMMON_ARGS+=("trainers.limit_num_batch=${LIMIT_NUM_BATCH}")
fi

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
        find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
            -name "${prefix}_*" -printf '%T@ %p\n' | sort -nr | \
            sed -n '1s/^[^ ]* //p'
    )"
    if [[ -z "${run_dir}" || ! -d "${run_dir}/checkpoints_every_epoch" ]]; then
        echo "cannot locate completed run for ${prefix}" >&2
        return 1
    fi

    checkpoint_dir="$(
        find "${run_dir}/checkpoints_every_epoch" -mindepth 1 -maxdepth 1 \
            -type d -name 'epoch:*' -print | sort -V | tail -n 1
    )"
    if [[ ! -f "${checkpoint_dir}/model.pt" || \
          ! -f "${checkpoint_dir}/classifier_rank0.pt" ]]; then
        echo "incomplete checkpoint: ${checkpoint_dir}" >&2
        return 1
    fi
    printf '%s\n' "${checkpoint_dir}"
}

echo "reusing stage 1 checkpoint: ${S1_CKPT}"

run_qgface() {
    local contrast_weight=1.0
    local s2_prefix="qgface_subcenter_s2_body36_0605"
    local s3_prefix="qgface_subcenter_s3_classifier_0605"
    local s4_prefix="qgface_subcenter_s4_joint_0605"
    local s2_ckpt
    local s3_ckpt

    run_train \
        "trainers.prefix=${s2_prefix}" \
        "trainers.batch_size=${S2_BATCH_SIZE}" \
        models=iresnet/configs/qgface_ir101_pretrained.yaml \
        "models.start_from=${S1_CKPT}/model.pt" \
        classifiers=configs/pfc40_subcenter_k3_freeze.yaml \
        "classifiers.start_from=${S1_CKPT}" \
        pefts=configs/part_freeze.yaml \
        pefts.target_modules=body.36 \
        "pipelines.contrast_weight=${contrast_weight}" \
        optims=configs/qgface_model_realign_sgd.yaml \
        optims.lr=0.008 \
        optims.num_epoch=15
    s2_ckpt="$(latest_checkpoint "${s2_prefix}")"
    echo "stage 2 checkpoint: ${s2_ckpt}"

    run_train \
        "trainers.prefix=${s3_prefix}" \
        "trainers.batch_size=${S3_BATCH_SIZE}" \
        models=iresnet/configs/qgface_ir101_pretrained.yaml \
        "models.start_from=${s2_ckpt}/model.pt" \
        models.freeze=true \
        classifiers=configs/pfc40_subcenter_k3.yaml \
        "classifiers.start_from=${s2_ckpt}" \
        pipelines.contrast_weight=0.0 \
        optims=configs/qgface_classifier_realign_sgd.yaml \
        optims.lr=0.006 \
        optims.classifier_lr=0.006
    s3_ckpt="$(latest_checkpoint "${s3_prefix}")"
    echo "stage 3 checkpoint: ${s3_ckpt}"

    run_train \
        "trainers.prefix=${s4_prefix}" \
        "trainers.batch_size=${S4_BATCH_SIZE}" \
        models=iresnet/configs/qgface_ir101_pretrained.yaml \
        "models.start_from=${s3_ckpt}/model.pt" \
        models.freeze=false \
        classifiers=configs/pfc40_subcenter_k3.yaml \
        "classifiers.start_from=${s3_ckpt}" \
        "pipelines.contrast_weight=${contrast_weight}" \
        optims=configs/qgface_model_realign_sgd.yaml \
        optims.lr=0.0008 \
        optims.classifier_lr=0.0002 \
        optims.num_epoch=15

    echo "QGFace stages 2-4 completed using the reused stage 1 checkpoint"
    echo "final checkpoint: $(latest_checkpoint "${s4_prefix}")"
}

run_qgface

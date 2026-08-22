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
S3_PREFIX="${S3_PREFIX:-qgface_subcenter_s3_classifier_0605}"
S4_PREFIX="${S4_PREFIX:-qgface_subcenter_s4_model_only_0605}"
S3_CKPT="${S3_CKPT:-}"
S4_BATCH_SIZE="${S4_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
EVALUATIONS_CONFIG="${EVALUATIONS_CONFIG:-configs/val_20260605.yaml}"

if [[ ! -x "${FABRIC_BIN}" ]]; then
    echo "fabric executable not found: ${FABRIC_BIN}" >&2
    exit 2
fi

latest_checkpoint() {
    local prefix="$1"
    local run_dir
    local checkpoint_dir

    run_dir="$({
        find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
            -name "${prefix}_*" -printf '%T@ %p\n'
    } | sort -nr | sed -n '1s/^[^ ]* //p')"
    if [[ -z "${run_dir}" || ! -d "${run_dir}/checkpoints_every_epoch" ]]; then
        echo "cannot locate completed run for ${prefix} under ${OUTPUT_ROOT}" >&2
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

if [[ -z "${S3_CKPT}" ]]; then
    S3_CKPT="$(latest_checkpoint "${S3_PREFIX}")"
fi
if [[ ! -f "${S3_CKPT}/model.pt" || \
      ! -f "${S3_CKPT}/classifier_rank0.pt" ]]; then
    echo "incomplete S3 checkpoint: ${S3_CKPT}" >&2
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

echo "reusing S3 checkpoint: ${S3_CKPT}"
"${FABRIC_BIN}" run \
    "--devices=${NUM_GPUS}" \
    --precision=bf16-mixed \
    train_opt.py \
    "${COMMON_ARGS[@]}" \
    "trainers.prefix=${S4_PREFIX}" \
    "trainers.batch_size=${S4_BATCH_SIZE}" \
    models=iresnet/configs/qgface_ir101_pretrained.yaml \
    "models.start_from=${S3_CKPT}/model.pt" \
    models.freeze=false \
    classifiers=configs/pfc40_subcenter_k3_freeze.yaml \
    "classifiers.start_from=${S3_CKPT}" \
    "pipelines.contrast_weight=1.0" \
    optims=configs/qgface_model_realign_sgd.yaml \
    optims.lr=0.0008 \
    optims.num_epoch=15

echo "S4 model-only training completed from ${S3_CKPT}"

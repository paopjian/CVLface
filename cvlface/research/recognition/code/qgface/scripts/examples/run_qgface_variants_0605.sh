#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ ! "${MODE}" =~ ^[1-6]$ ]]; then
    echo "Usage: bash scripts/examples/run_qgface_variants_0605.sh <1|2|3|4|5|6>"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="${DECODE_BACKEND:-turbojpeg}"
export DATA_ROOT="${DATA_ROOT:-/data1/dataset_0605}"
export PYTHONUNBUFFERED=1

if [[ "${MODE}" == "6" ]]; then
    if [[ -n "${BATCH_SIZE:-}" && "${BATCH_SIZE}" != "128" ]]; then
        echo "Mode 6 forces per-GPU batch size to 128; ignoring BATCH_SIZE=${BATCH_SIZE}"
    fi
    BATCH_SIZE=128
else
    BATCH_SIZE="${BATCH_SIZE:-64}"
fi
NUM_WORKERS="${NUM_WORKERS:-8}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
EVALUATIONS_CONFIG="${EVALUATIONS_CONFIG:-configs/val_20260605.yaml}"
OUTPUT_ROOT="/data1/dataset_0605/train_output"

COMMON_OVERRIDES=(
    trainers.num_gpu=8
    "trainers.batch_size=${BATCH_SIZE}"
    "trainers.num_workers=${NUM_WORKERS}"
    trainers.precision=bf16-mixed
    trainers.using_wandb=true
    "trainers.compile_model=${COMPILE_MODEL}"
    trainers.channels_last=true
    trainers.skip_final_eval=true
    dataset=configs/dataset_0605_train_rec.yaml
    data_augs=configs/qgface.yaml
    losses=configs/qgface_adaface.yaml
    pipelines=configs/train_qgface.yaml
    "evaluations=${EVALUATIONS_CONFIG}"
    "dataset.model_save_dir=${OUTPUT_ROOT}"
)

if [[ -n "${LIMIT_NUM_BATCH:-}" ]]; then
    COMMON_OVERRIDES+=("trainers.limit_num_batch=${LIMIT_NUM_BATCH}")
fi

RUNTIME_OVERRIDES=()
if [[ -n "${NUM_EPOCH:-}" ]]; then
    if [[ "${MODE}" == "5" || "${MODE}" == "6" ]]; then
        echo "NUM_EPOCH cannot override the fixed multi-stage schedule of mode ${MODE}"
        exit 2
    fi
    RUNTIME_OVERRIDES+=("optims.num_epoch=${NUM_EPOCH}")
fi
if [[ "${MODE}" == "6" && ( -n "${LR:-}" || -n "${CLASSIFIER_LR:-}" ) ]]; then
    echo "Mode 6 uses fixed per-stage learning rates; unset LR and CLASSIFIER_LR"
    exit 2
fi
if [[ -n "${LR:-}" ]]; then
    RUNTIME_OVERRIDES+=("optims.lr=${LR}")
fi
if [[ -n "${CLASSIFIER_LR:-}" ]]; then
    RUNTIME_OVERRIDES+=("optims.classifier_lr=${CLASSIFIER_LR}")
fi

run_qgface() {
    echo "Starting QGFace mode ${MODE}: GPUs=${CUDA_VISIBLE_DEVICES}, batch_size=${BATCH_SIZE}, model_lr=${LR:-config_default}, classifier_lr=${CLASSIFIER_LR:-config_default}, compile=${COMPILE_MODEL}"
    conda run --no-capture-output -n cvlface fabric run --devices=8 --precision=bf16-mixed \
        train_opt.py "${COMMON_OVERRIDES[@]}" "$@" "${RUNTIME_OVERRIDES[@]}"
}

case "${MODE}" in
    1)
        run_qgface \
            trainers.prefix=qgface_1_ir34_sharded_fc \
            models=iresnet/configs/qgface_ir34.yaml \
            classifiers=configs/sharded_fc.yaml \
            optims=configs/qgface_sgd.yaml
        ;;
    2)
        run_qgface \
            trainers.prefix=qgface_2_ir34_pfc40 \
            models=iresnet/configs/qgface_ir34.yaml \
            classifiers=configs/pfc40.yaml \
            optims=configs/qgface_sgd.yaml
        ;;
    3)
        run_qgface \
            trainers.prefix=qgface_3_ir101_pfc40 \
            models=iresnet/configs/qgface_ir101.yaml \
            classifiers=configs/pfc40.yaml \
            optims=configs/qgface_sgd.yaml
        ;;
    4)
        run_qgface \
            trainers.prefix=qgface_4_ir101_pretrained_pfc40 \
            models=iresnet/configs/qgface_ir101_pretrained.yaml \
            classifiers=configs/pfc40.yaml \
            optims=configs/qgface_pretrained_joint_sgd.yaml
        ;;
    5)
        STAGE1_PREFIX="qgface_5_ir101_pretrained_classifier_warmup"
        if [[ -z "${STAGE1_CKPT:-}" ]]; then
            run_qgface \
                "trainers.prefix=${STAGE1_PREFIX}" \
                models=iresnet/configs/qgface_ir101_pretrained.yaml \
                models.freeze=true \
                classifiers=configs/pfc40.yaml \
                pipelines.contrast_weight=0.0 \
                optims=configs/qgface_classifier_warmup_sgd.yaml

            STAGE1_RUN="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
                -name "${STAGE1_PREFIX}_*" -printf '%T@ %p\n' | sort -nr | \
                sed -n '1s/^[^ ]* //p')"
            if [[ -z "${STAGE1_RUN}" || ! -d "${STAGE1_RUN}/checkpoints_every_epoch" ]]; then
                echo "Cannot locate stage-1 run under ${OUTPUT_ROOT}"
                exit 1
            fi
            STAGE1_CKPT="$(find "${STAGE1_RUN}/checkpoints_every_epoch" \
                -mindepth 1 -maxdepth 1 -type d -name 'epoch:4_step:*' -print -quit)"
        fi

        if [[ ! -f "${STAGE1_CKPT}/classifier_rank0.pt" ]]; then
            echo "Invalid stage-1 checkpoint: ${STAGE1_CKPT}"
            exit 1
        fi

        run_qgface \
            trainers.prefix=qgface_5_ir101_pretrained_model_train \
            models=iresnet/configs/qgface_ir101_pretrained.yaml \
            classifiers=configs/pfc40_freeze.yaml \
            "classifiers.start_from=${STAGE1_CKPT}" \
            optims=configs/qgface_model_finetune_sgd.yaml
        ;;
    6)
        STAGE6_CLASSIFIER_PREFIX="qgface_6_ir101_epoch11_classifier_realign"
        if [[ -z "${STAGE6_CLASSIFIER_CKPT:-}" ]]; then
            if [[ -z "${MODE5_EPOCH11_CKPT:-}" ]]; then
                MODE5_RUN="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
                    -name 'qgface_5_ir101_pretrained_model_train_*' -printf '%T@ %p\n' | \
                    sort -nr | sed -n '1s/^[^ ]* //p')"
                if [[ -z "${MODE5_RUN}" || ! -d "${MODE5_RUN}/checkpoints_every_epoch" ]]; then
                    echo "Cannot locate mode-5 model run under ${OUTPUT_ROOT}"
                    exit 1
                fi
                MODE5_EPOCH11_CKPT="$(find "${MODE5_RUN}/checkpoints_every_epoch" \
                    -mindepth 1 -maxdepth 1 -type d -name 'epoch:11_step:*' -print -quit)"
            fi

            if [[ ! -f "${MODE5_EPOCH11_CKPT}/model.pt" || \
                  ! -f "${MODE5_EPOCH11_CKPT}/classifier_rank0.pt" ]]; then
                echo "Invalid mode-5 epoch-11 checkpoint: ${MODE5_EPOCH11_CKPT}"
                exit 1
            fi

            run_qgface \
                "trainers.prefix=${STAGE6_CLASSIFIER_PREFIX}" \
                models=iresnet/configs/qgface_ir101_pretrained.yaml \
                "models.start_from=${MODE5_EPOCH11_CKPT}/model.pt" \
                models.freeze=true \
                classifiers=configs/pfc40.yaml \
                "classifiers.start_from=${MODE5_EPOCH11_CKPT}" \
                pipelines.contrast_weight=0.0 \
                optims=configs/qgface_classifier_realign_sgd.yaml

            STAGE6_CLASSIFIER_RUN="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
                -name "${STAGE6_CLASSIFIER_PREFIX}_*" -printf '%T@ %p\n' | sort -nr | \
                sed -n '1s/^[^ ]* //p')"
            if [[ -z "${STAGE6_CLASSIFIER_RUN}" || \
                  ! -d "${STAGE6_CLASSIFIER_RUN}/checkpoints_every_epoch" ]]; then
                echo "Cannot locate mode-6 classifier run under ${OUTPUT_ROOT}"
                exit 1
            fi
            STAGE6_CLASSIFIER_CKPT="$(find "${STAGE6_CLASSIFIER_RUN}/checkpoints_every_epoch" \
                -mindepth 1 -maxdepth 1 -type d -name 'epoch:4_step:*' -print -quit)"
        fi

        if [[ ! -f "${STAGE6_CLASSIFIER_CKPT}/model.pt" || \
              ! -f "${STAGE6_CLASSIFIER_CKPT}/classifier_rank0.pt" ]]; then
            echo "Invalid mode-6 classifier checkpoint: ${STAGE6_CLASSIFIER_CKPT}"
            exit 1
        fi

        run_qgface \
            trainers.prefix=qgface_6_ir101_epoch11_model_realign \
            models=iresnet/configs/qgface_ir101_pretrained.yaml \
            "models.start_from=${STAGE6_CLASSIFIER_CKPT}/model.pt" \
            models.freeze=false \
            classifiers=configs/pfc40_freeze.yaml \
            "classifiers.start_from=${STAGE6_CLASSIFIER_CKPT}" \
            optims=configs/qgface_model_realign_sgd.yaml
        ;;
esac

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

# 自动断点续传：1=启用(默认), 0=每个 stage 都从头训练
AUTO_RESUME="${AUTO_RESUME:-1}"

PREFIX="subcenter_id_enhance"
S1_PREFIX="${PREFIX}_s1_cls04"
S2_PREFIX="${PREFIX}_s2_body36"
S3_PREFIX="${PREFIX}_s3_cls04"
S4_PREFIX="${PREFIX}_s4_joint04"

# 各 stage 的总 epoch 数，用于判断 stage 是否已训练完成
S1_EPOCH=5
S2_EPOCH=15
S3_EPOCH=5
S4_EPOCH=15

if [[ ! "${START_STAGE}" =~ ^[1-4]$ ]]; then
    echo "START_STAGE must be an integer from 1 to 4, got: ${START_STAGE}" >&2
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
    "dataset=configs/dataset_0605_train_id_enhance.yaml"
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

# 判断一个 epoch 目录是否含有完整的续训状态
# 续训需要 pipeline.pt / model.pt / optimizer.pt / lr_scheduler.pt 以及全部 rank 的 classifier
is_resumable_checkpoint() {
    local ckpt="$1"
    local f rank

    for f in pipeline.pt model.pt optimizer.pt lr_scheduler.pt; do
        [[ -f "${ckpt}/${f}" ]] || return 1
    done
    for (( rank = 0; rank < NUM_GPUS; rank++ )); do
        [[ -f "${ckpt}/classifier_rank${rank}.pt" ]] || return 1
    done
    return 0
}

# 扫描某 prefix 下所有 run 目录，按 "epoch 号" 输出 "<epoch> <路径>"
# require_resumable=1 时只输出状态完整、可用于续训的 checkpoint
list_epoch_checkpoints() {
    local prefix="$1"
    local require_resumable="${2:-0}"
    local run_dir ckpt epoch_num

    while IFS= read -r run_dir; do
        [[ -d "${run_dir}/checkpoints_every_epoch" ]] || continue
        while IFS= read -r ckpt; do
            epoch_num="$(basename "${ckpt}")"
            epoch_num="${epoch_num#epoch:}"
            [[ "${epoch_num}" =~ ^[0-9]+$ ]] || continue
            if (( require_resumable )) && ! is_resumable_checkpoint "${ckpt}"; then
                continue
            fi
            printf '%s %s\n' "${epoch_num}" "${ckpt}"
        done < <(
            find "${run_dir}/checkpoints_every_epoch" -mindepth 1 -maxdepth 1 \
                -type d -name 'epoch:*' -print 2>/dev/null
        )
    done < <(
        find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" \
            -print 2>/dev/null
    )
}

# 取 epoch 号最大的 checkpoint（跨同 prefix 的多个 run 目录）
# 同 epoch 号时取修改时间最新的
pick_max_epoch_checkpoint() {
    local prefix="$1"
    local require_resumable="${2:-0}"

    list_epoch_checkpoints "${prefix}" "${require_resumable}" \
        | while read -r epoch_num ckpt; do
              printf '%s %s %s\n' "${epoch_num}" "$(stat -c '%Y' "${ckpt}")" "${ckpt}"
          done \
        | sort -k1,1n -k2,2n \
        | tail -n 1 \
        | cut -d ' ' -f 3-
}

# stage 交接用：拿该 stage 训练到的最后一个 checkpoint
latest_checkpoint() {
    local prefix="$1"
    local checkpoint_dir

    checkpoint_dir="$(pick_max_epoch_checkpoint "${prefix}" 0)"
    if [[ -z "${checkpoint_dir}" ]]; then
        echo "no epoch checkpoint found for prefix ${prefix} under ${OUTPUT_ROOT}" >&2
        return 1
    fi
    if [[ ! -f "${checkpoint_dir}/model.pt" || ! -f "${checkpoint_dir}/classifier_rank0.pt" ]]; then
        echo "incomplete checkpoint: ${checkpoint_dir}" >&2
        return 1
    fi

    printf '%s\n' "${checkpoint_dir}"
}

# 自动断点探测。输出结果写入全局变量：
#   RESUME_ARGS  —— 传给 train_opt.py 的 resume 参数数组（可能为空）
#   STAGE_DONE   —— 1 表示该 stage 已训练完成，可整段跳过
detect_resume() {
    local stage="$1"
    local prefix="$2"
    local num_epoch="$3"
    local ckpt last_epoch

    RESUME_ARGS=()
    STAGE_DONE=0

    if (( ! AUTO_RESUME )); then
        echo "[stage ${stage}] AUTO_RESUME=0, 从头开始训练"
        return 0
    fi

    ckpt="$(pick_max_epoch_checkpoint "${prefix}" 1)"
    if [[ -z "${ckpt}" ]]; then
        echo "[stage ${stage}] 未找到可续训的 checkpoint, 从头开始训练"
        return 0
    fi

    last_epoch="$(basename "${ckpt}")"
    last_epoch="${last_epoch#epoch:}"

    if (( last_epoch >= num_epoch - 1 )); then
        echo "[stage ${stage}] 已完成 (最后 epoch=${last_epoch}/$(( num_epoch - 1 ))), 跳过训练"
        echo "[stage ${stage}] checkpoint: ${ckpt}"
        STAGE_DONE=1
        return 0
    fi

    echo "[stage ${stage}] 断点续传: 从 epoch ${last_epoch} 恢复, 将继续训练 epoch $(( last_epoch + 1 ))..$(( num_epoch - 1 ))"
    echo "[stage ${stage}] resume: ${ckpt}"
    RESUME_ARGS=("trainers.resume=${ckpt}")
    return 0
}

echo "================ 断点续传扫描 ================"
echo "AUTO_RESUME=${AUTO_RESUME}  START_STAGE=${START_STAGE}  NUM_GPUS=${NUM_GPUS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "============================================="

if (( START_STAGE <= 1 )); then
    detect_resume 1 "${S1_PREFIX}" "${S1_EPOCH}"
    if (( ! STAGE_DONE )); then
        run_train \
            "trainers.prefix=${S1_PREFIX}" \
            "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
            trainers.batch_size=512 \
            trainers.skip_final_eval=True \
            "models.start_from=${PRETRAINED_MODEL}" \
            models.freeze=True \
            classifiers.freeze=False \
            classifiers.sample_rate=0.4 \
            data_augs=configs/basic_v2_numpy.yaml \
            pefts=configs/freeze.yaml \
            optims.lr=0.008 \
            "optims.num_epoch=${S1_EPOCH}" \
            "optims.lr_milestones=[2,4]" \
            optims.weight_decay=0.0001 \
            optims.lr_lambda=0.3
    fi
fi
S1_CKPT="$(latest_checkpoint "${S1_PREFIX}")"
echo "stage 1 checkpoint: ${S1_CKPT}"

if (( START_STAGE <= 2 )); then
    detect_resume 2 "${S2_PREFIX}" "${S2_EPOCH}"
    if (( ! STAGE_DONE )); then
        run_train \
            "trainers.prefix=${S2_PREFIX}" \
            "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
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
            "optims.num_epoch=${S2_EPOCH}" \
            optims.warmup_epoch=2 \
            optims.weight_decay=0.0005 \
            optims.lr_lambda=0.1 \
            optims.scheduler=cosine
    fi
fi
S2_CKPT="$(latest_checkpoint "${S2_PREFIX}")"
echo "stage 2 checkpoint: ${S2_CKPT}"

if (( START_STAGE <= 3 )); then
    detect_resume 3 "${S3_PREFIX}" "${S3_EPOCH}"
    if (( ! STAGE_DONE )); then
        run_train \
            "trainers.prefix=${S3_PREFIX}" \
            "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
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
            "optims.num_epoch=${S3_EPOCH}" \
            "optims.lr_milestones=[2,4]" \
            optims.weight_decay=0.0001 \
            optims.lr_lambda=0.3
    fi
fi
S3_CKPT="$(latest_checkpoint "${S3_PREFIX}")"
echo "stage 3 checkpoint: ${S3_CKPT}"

if (( START_STAGE <= 4 )); then
    detect_resume 4 "${S4_PREFIX}" "${S4_EPOCH}"
    if (( ! STAGE_DONE )); then
        run_train \
            "trainers.prefix=${S4_PREFIX}" \
            "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
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
            "optims.num_epoch=${S4_EPOCH}" \
            optims.warmup_epoch=2 \
            optims.weight_decay=0.0005 \
            optims.lr_lambda=0.1 \
            optims.scheduler=cosine
    fi
fi
S4_CKPT="$(latest_checkpoint "${S4_PREFIX}")"
echo "stage 4 checkpoint: ${S4_CKPT}"

echo "four-stage training completed"
echo "final model checkpoint: ${S4_CKPT}/model.pt"
echo "classifier for dataset cleaning: ${S4_CKPT}"

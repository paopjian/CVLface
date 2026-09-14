#!/usr/bin/env bash
################################################################################
# TopoFR dataset_0605 四阶段训练
#
# 阶段 1: 冻结 backbone，训练 PartialFC 分类器
# 阶段 2: 解冻 body.36 之后的 backbone，冻结分类器
# 阶段 3: 冻结 backbone，继续训练 PartialFC 分类器
# 阶段 4: 解冻整个 backbone，冻结 PartialFC 分类器
#
# 脚本可重复执行：
# - 自动搜索最新 Glint360K_R100_TopoFR 预训练权重；
# - 自动复用已完成阶段的最新 checkpoint；
# - 阶段中断后自动通过 trainers.resume 断点续训；
# - 不需要手动输入 checkpoint 路径。
################################################################################

set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}"
export DECODE_BACKEND="turbojpeg"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source "/root/anaconda3/bin/activate" cvlface
cd "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr"

DATA_ROOT="${DATA_ROOT:-/data1/dataset_0605}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/train_output}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-True}"

# 可通过环境变量覆盖阶段前缀，默认前缀保持与旧脚本兼容。
STAGE1_PREFIX="${STAGE1_PREFIX:-s1_topofr_warmup_0605}"
STAGE2_PREFIX="${STAGE2_PREFIX:-s2_topofr_body36_0605}"
STAGE3_PREFIX="${STAGE3_PREFIX:-s3_topofr_classifier_0605}"
STAGE4_PREFIX="${STAGE4_PREFIX:-s4_topofr_full_0605}"

if [[ -n "${PRETRAINED_MODEL:-}" ]]; then
    PRETRAINED_MODEL="${PRETRAINED_MODEL}"
else
    PRETRAINED_MODEL="$({
        find "$PRETRAINED_ROOT" -maxdepth 1 -type f \
            -name 'Glint360K_R100_TopoFR_*.pt' -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}')"
fi

if [[ -z "$PRETRAINED_MODEL" || ! -f "$PRETRAINED_MODEL" ]]; then
    echo "错误：未找到 Glint360K_R100_TopoFR 预训练模型。"
    echo "搜索目录：$PRETRAINED_ROOT"
    echo "可通过 PRETRAINED_MODEL=/path/to/model.pt 覆盖。"
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "错误：数据目录不存在：$DATA_ROOT"
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

# checkpoints_every_epoch/<epoch:N_step:M> 必须同时存在 pipeline/model，
# 否则视为未完成 checkpoint，允许下次运行继续恢复。
latest_checkpoint() {
    local prefix="$1"
    find "$OUTPUT_ROOT" -type f \
        -path "*/${prefix}_*/checkpoints_every_epoch/*/pipeline.pt" \
        -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}'
}

checkpoint_epoch() {
    local checkpoint="$1"
    basename "$checkpoint" | sed -n 's/^epoch:\([0-9][0-9]*\).*$/\1/p'
}

checkpoint_is_complete() {
    local checkpoint="$1"
    [[ -n "$checkpoint" && -f "$checkpoint/pipeline.pt" && -f "$checkpoint/model.pt" ]]
}

stage_is_complete() {
    local checkpoint="$1"
    local expected_last_epoch="$2"
    checkpoint_is_complete "$checkpoint" || return 1
    local epoch
    epoch="$(checkpoint_epoch "$checkpoint")"
    [[ "$epoch" =~ ^[0-9]+$ && "$epoch" -ge "$expected_last_epoch" ]]
}

# 所有阶段使用同一个 TopoFR backbone 和 PartialFC；阶段命令再覆盖
# models.start_from、models.freeze、classifiers.freeze 等阶段特有选项。
# 阶段 2 保留 models.freeze=True：TopoFR pipeline 会据此只让 part_freeze
# 解冻范围内的 BN 进入 train 模式，避免冻结部分的 running statistics 被更新。
COMMON_ARGS=(
    "trainers.num_gpu=8"
    "trainers.batch_size=128"
    "trainers.num_workers=8"
    "trainers.precision=bf16-mixed"
    "models=iresnet_insightface/configs/v1_ir101.yaml"
    "models.start_from=${PRETRAINED_MODEL}"
    "dataset=configs/dataset_0605_train_rec.yaml"
    "dataset.model_save_dir=${OUTPUT_ROOT}"
    "data_augs=configs/basic_v2_numpy.yaml"
    "classifiers=configs/partial_fc.yaml"
    "classifiers.sample_rate=0.40"
    "losses=configs/topofr_adaface.yaml"
    "pipelines=configs/train_model_cls_topofr"
    "evaluations=configs/val_20260605.yaml"
)

run_stage() {
    local prefix="$1"
    local resume_checkpoint="$2"
    shift 2
    local resume_arg=()
    if [[ -n "$resume_checkpoint" ]]; then
        resume_arg=("trainers.resume=${resume_checkpoint}")
        echo "断点续训：${resume_checkpoint}"
    fi

    fabric run --devices=8 --precision="bf16-mixed" \
        train_opt.py \
        "trainers.prefix=${prefix}" \
        "trainers.skip_final_eval=${SKIP_FINAL_EVAL}" \
        "${COMMON_ARGS[@]}" \
        "${resume_arg[@]}" \
        "$@"
}

run_or_resume_stage() {
    local stage_name="$1"
    local prefix="$2"
    local expected_last_epoch="$3"
    shift 3

    local checkpoint
    STAGE_RESULT=""
    checkpoint="$(latest_checkpoint "$prefix")"
    if stage_is_complete "$checkpoint" "$expected_last_epoch"; then
        echo "${stage_name} 已完成，复用最新 checkpoint：${checkpoint}"
        STAGE_RESULT="$checkpoint"
        return 0
    fi

    echo "========================================="
    echo "${stage_name}"
    echo "========================================="
    if [[ -n "$checkpoint" ]]; then
        echo "${stage_name} 存在未完成 checkpoint：${checkpoint}"
    else
        echo "${stage_name} 未找到 checkpoint，从头开始"
    fi
    run_stage "$prefix" "$checkpoint" "$@"

    checkpoint="$(latest_checkpoint "$prefix")"
    if ! stage_is_complete "$checkpoint" "$expected_last_epoch"; then
        echo "错误：${stage_name} 结束后未找到完整 checkpoint。"
        echo "检查目录：${OUTPUT_ROOT}"
        return 1
    fi
    echo "${stage_name} 完成：${checkpoint}"
    STAGE_RESULT="$checkpoint"
}

echo "TopoFR dataset_0605 四阶段训练（8 卡）"
echo "预训练模型：${PRETRAINED_MODEL}"
echo "分类器：PartialFC（sample_rate=0.40）"
echo "输出目录：${OUTPUT_ROOT}"

# num_epoch=N 的最后一个保存 epoch 是 N-1。
echo "[阶段1配置] 输入模型=${PRETRAINED_MODEL}; lr=0.008; epoch=5; 训练=PartialFC，backbone冻结"
run_or_resume_stage \
    "阶段1：PartialFC warmup（backbone 冻结）" \
    "$STAGE1_PREFIX" 4 \
    "models.freeze=True" \
    "pefts=configs/freeze.yaml" \
    "optims=configs/step_sgd.yaml" \
    "optims.lr=0.008" \
    "optims.num_epoch=5" \
    "optims.lr_milestones=[2,4]" \
    "optims.momentum=0.9" \
    "optims.weight_decay=0.0001" \
    "optims.lr_lambda=0.3" \
    "optims.max_grad_norm=5.0"
STAGE1_CKPT="$STAGE_RESULT"

echo "[阶段1输出] checkpoint=${STAGE1_CKPT}"
echo "[阶段2配置] 输入模型=${STAGE1_CKPT}/model.pt; lr=0.008; epoch=15; 训练=body.36之后的backbone，PartialFC冻结，其他BN统计冻结"
run_or_resume_stage \
    "阶段2：body.36 解冻（PartialFC 冻结）" \
    "$STAGE2_PREFIX" 14 \
    "models.start_from=${STAGE1_CKPT}/model.pt" \
    "models.freeze=True" \
    "classifiers.freeze=True" \
    "data_augs=configs/gridsample_v2_numpy.yaml" \
    "pefts=configs/part_freeze.yaml" \
    "pefts.target_modules=body.36" \
    "pefts.classifier_ckpt_dir=${STAGE1_CKPT}" \
    "optims=configs/step_sgd.yaml" \
    "optims.lr=0.008" \
    "optims.num_epoch=15" \
    "optims.warmup_epoch=2" \
    "optims.momentum=0.9" \
    "optims.weight_decay=0.0005" \
    "optims.lr_lambda=0.1" \
    "optims.max_grad_norm=5.0" \
    "optims.scheduler=cosine"
STAGE2_CKPT="$STAGE_RESULT"

echo "[阶段2输出] checkpoint=${STAGE2_CKPT}"
echo "[阶段3配置] 输入模型=${STAGE2_CKPT}/model.pt; lr=0.006; epoch=5; 训练=PartialFC，backbone冻结"
run_or_resume_stage \
    "阶段3：PartialFC 重训练（backbone 冻结）" \
    "$STAGE3_PREFIX" 4 \
    "models.start_from=${STAGE2_CKPT}/model.pt" \
    "models.freeze=True" \
    "data_augs=configs/gridsample_v2_numpy.yaml" \
    "pefts=configs/freeze.yaml" \
    "pefts.classifier_ckpt_dir=${STAGE2_CKPT}" \
    "optims=configs/step_sgd.yaml" \
    "optims.lr=0.006" \
    "optims.num_epoch=5" \
    "optims.lr_milestones=[2,4]" \
    "optims.momentum=0.9" \
    "optims.weight_decay=0.0001" \
    "optims.lr_lambda=0.3" \
    "optims.max_grad_norm=5.0"
STAGE3_CKPT="$STAGE_RESULT"

echo "[阶段3输出] checkpoint=${STAGE3_CKPT}"
echo "[阶段4配置] 输入模型=${STAGE3_CKPT}/model.pt; lr=0.0008; epoch=15; 训练=全backbone，PartialFC冻结"
run_or_resume_stage \
    "阶段4：全模型微调（PartialFC 冻结）" \
    "$STAGE4_PREFIX" 14 \
    "models.start_from=${STAGE3_CKPT}/model.pt" \
    "models.freeze=False" \
    "classifiers.freeze=True" \
    "data_augs=configs/gridsample_v2_numpy.yaml" \
    "pefts=configs/full.yaml" \
    "pefts.classifier_ckpt_dir=${STAGE3_CKPT}" \
    "optims=configs/step_sgd.yaml" \
    "optims.lr=0.0008" \
    "optims.num_epoch=15" \
    "optims.warmup_epoch=2" \
    "optims.momentum=0.9" \
    "optims.weight_decay=0.0005" \
    "optims.lr_lambda=0.1" \
    "optims.max_grad_norm=5.0" \
    "optims.scheduler=cosine"

echo "========================================="
echo "TopoFR 四阶段训练完成"
echo "阶段1 checkpoint：${STAGE1_CKPT}"
echo "阶段2 checkpoint：${STAGE2_CKPT}"
echo "阶段3 checkpoint：${STAGE3_CKPT}"
echo "阶段4 checkpoint：$(latest_checkpoint "$STAGE4_PREFIX")"
echo "========================================="

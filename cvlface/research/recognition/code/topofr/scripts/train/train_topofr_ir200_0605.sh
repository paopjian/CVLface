#!/usr/bin/env bash
################################################################################
# TopoFR IR-200 dataset_0605 四阶段训练
#
# 阶段 1: 冻结 backbone，训练 PartialFC 分类器
# 阶段 2: 解冻 layer3.48 之后的 backbone，冻结分类器
# 阶段 3: 冻结 backbone，继续训练 PartialFC 分类器
# 阶段 4: 解冻整个 backbone，冻结 PartialFC 分类器
#
# 脚本可重复执行：
# - 自动加载 Glint360K_R200_TopoFR 预训练权重；
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
PRETRAINED_ROOT="${PRETRAINED_ROOT:-/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-True}"

# train_opt.py 的 checkpoint 由 config.py:prepare_output_dir 固定写到
# research/experiments/<task>/<prefix>_<date>_<trial>/checkpoints/，与
# dataset.model_save_dir 无关（后者只被 train.py/train5.py 使用）。
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/root/zhaokj/CVLface/cvlface/research/recognition/experiments/topofr}"

# 可通过环境变量覆盖阶段前缀，默认前缀保持与旧脚本兼容。
STAGE1_PREFIX="${STAGE1_PREFIX:-s1_topofr_ir200_warmup_0605}"
STAGE2_PREFIX="${STAGE2_PREFIX:-s2_topofr_ir200_body72_0605}"
STAGE3_PREFIX="${STAGE3_PREFIX:-s3_topofr_ir200_classifier_0605}"
STAGE4_PREFIX="${STAGE4_PREFIX:-s4_topofr_ir200_full_0605}"

if [[ -n "${PRETRAINED_MODEL:-}" ]]; then
    PRETRAINED_MODEL="${PRETRAINED_MODEL}"
else
    PRETRAINED_MODEL="$({
        find "$PRETRAINED_ROOT" -maxdepth 1 -type f \
            -name 'Glint360K_R200_TopoFR_*.pt' -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}')"
fi

if [[ -z "$PRETRAINED_MODEL" || ! -f "$PRETRAINED_MODEL" ]]; then
    echo "错误：未找到 Glint360K_R200_TopoFR 预训练模型。"
    echo "搜索目录：$PRETRAINED_ROOT"
    echo "可通过 PRETRAINED_MODEL=/path/to/model.pt 覆盖。"
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "错误：数据目录不存在：$DATA_ROOT"
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

# checkpoints/<epoch:N_step:M> 必须同时存在 model/pipeline/rank0 分类器分片，
# 否则视为未完成 checkpoint（例如 rank0 未参与保存时留下的残骸目录）。
checkpoint_is_complete() {
    local checkpoint="$1"
    [[ -n "$checkpoint" \
        && -f "$checkpoint/model.pt" \
        && -f "$checkpoint/pipeline.pt" \
        && -f "$checkpoint/classifier_rank0.pt" ]]
}

# 返回该 prefix 下最新的**完整** checkpoint。按 mtime 倒序逐个校验，
# 跳过残缺目录，避免把早停时的半成品当成阶段产物。
#
# train_opt.py 有两条独立的保存路径，必须都搜：
#   A. {EXPERIMENT_ROOT}/<run>/checkpoints/epoch:N_step:M
#      —— train_pipeline.save()，位于 `model.has_trainable_params()` 分支内部。
#         backbone 全冻结的阶段（s1/s3）该分支整体跳过，因此不会有产物。
#   B. {OUTPUT_ROOT}/<run>/checkpoints_every_epoch/epoch:N
#      —— save_pipelines_and_configs()，在分支外每个 epoch 无条件执行，
#         所以这是冻结 backbone 阶段唯一的产物来源。
latest_checkpoint() {
    local prefix="$1"
    local candidate
    while IFS= read -r candidate; do
        [[ -z "$candidate" ]] && continue
        if checkpoint_is_complete "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <({
                find "$EXPERIMENT_ROOT" -maxdepth 3 -type d \
                    -path "*/${prefix}_*/checkpoints/epoch:*" \
                    -printf '%T@ %p\n' 2>/dev/null || true
                find "$OUTPUT_ROOT" -maxdepth 3 -type d \
                    -path "*/${prefix}_*/checkpoints_every_epoch/epoch:*" \
                    -printf '%T@ %p\n' 2>/dev/null || true
             } | sort -nr | sed 's/^[^ ]* //')
    return 0
}

checkpoint_epoch() {
    local checkpoint="$1"
    basename "$checkpoint" | sed -n 's/^epoch:\([0-9][0-9]*\).*$/\1/p'
}

# 阶段完成判定：只要存在完整 checkpoint 即视为完成。
# 不再要求 epoch 数达到 num_epoch-1 —— 早停是正常终止路径，
# 若强求跑满 epoch，早停后脚本会永远判定该阶段失败而无法进入下一阶段。
stage_is_complete() {
    local checkpoint="$1"
    checkpoint_is_complete "$checkpoint"
}

# 所有阶段使用同一个 TopoFR IR-200 backbone 和 PartialFC；阶段命令再覆盖
# models.start_from、models.freeze、classifiers.freeze 等阶段特有选项。
# 阶段 2 保留 models.freeze=True：TopoFR pipeline 会据此只让 part_freeze
# 解冻范围内的 BN 进入 train 模式，避免冻结部分的 running statistics 被更新。
COMMON_ARGS=(
    "trainers.num_gpu=8"
    "trainers.batch_size=128"
    "trainers.num_workers=8"
    "trainers.precision=bf16-mixed"
    "models=iresnet_insightface/configs/v1_ir200.yaml"
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
    shift 2

    local checkpoint
    STAGE_RESULT=""
    checkpoint="$(latest_checkpoint "$prefix")"
    if stage_is_complete "$checkpoint"; then
        echo "${stage_name} 已完成，复用最新 checkpoint：${checkpoint}"
        echo "  （epoch=$(checkpoint_epoch "$checkpoint")；早停提前结束亦视为完成）"
        STAGE_RESULT="$checkpoint"
        return 0
    fi

    echo "========================================="
    echo "${stage_name}"
    echo "========================================="
    echo "${stage_name} 未找到完整 checkpoint，从头开始"
    run_stage "$prefix" "" "$@"

    checkpoint="$(latest_checkpoint "$prefix")"
    if ! stage_is_complete "$checkpoint"; then
        echo "错误：${stage_name} 结束后未找到完整 checkpoint。"
        echo "检查目录：${EXPERIMENT_ROOT}/${prefix}_*/checkpoints/"
        return 1
    fi
    echo "${stage_name} 完成：${checkpoint}（epoch=$(checkpoint_epoch "$checkpoint")）"
    STAGE_RESULT="$checkpoint"
}

echo "TopoFR IR-200 dataset_0605 四阶段训练（8 卡）"
echo "预训练模型：${PRETRAINED_MODEL}"
echo "分类器：PartialFC（sample_rate=0.40）"
echo "输出目录：${OUTPUT_ROOT}"

# 上游阶段的 checkpoint 只被其下游阶段消费。若下游阶段已经产出完整
# checkpoint，则上游阶段无需重跑——即使它自身的目录已被清理。
# 用于避免"s2 已完成但 s1 产物丢失"时白跑一遍 s1。
STAGE2_DONE_CKPT="$(latest_checkpoint "$STAGE2_PREFIX")"
STAGE3_DONE_CKPT="$(latest_checkpoint "$STAGE3_PREFIX")"
STAGE4_DONE_CKPT="$(latest_checkpoint "$STAGE4_PREFIX")"

# num_epoch=N 的最后一个保存 epoch 是 N-1。
# IR-200 比 IR-100 更深，阶段 2 解冻点设为 body.72（IR-100 是 body.36，保持相同比例）
# body.72 对应 IR-200 的 layer1[6] + layer2[26] + layer3[40] = 从 layer3.40 开始解冻
if [[ -n "$STAGE2_DONE_CKPT" || -n "$STAGE3_DONE_CKPT" || -n "$STAGE4_DONE_CKPT" ]]; then
    echo "阶段1：跳过（下游阶段已有完整 checkpoint，其产物不再被需要）"
    STAGE1_CKPT=""
else
    echo "[阶段1配置] 输入模型=${PRETRAINED_MODEL}; lr=0.008; epoch=5; 训练=PartialFC，backbone冻结"
    run_or_resume_stage \
        "阶段1：PartialFC warmup（backbone 冻结）" \
        "$STAGE1_PREFIX" \
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
fi

if [[ -n "$STAGE2_DONE_CKPT" ]]; then
    echo "阶段2：已完成，复用 checkpoint：${STAGE2_DONE_CKPT}"
    echo "  （epoch=$(checkpoint_epoch "$STAGE2_DONE_CKPT")；早停提前结束亦视为完成）"
    STAGE2_CKPT="$STAGE2_DONE_CKPT"
elif [[ -n "$STAGE3_DONE_CKPT" || -n "$STAGE4_DONE_CKPT" ]]; then
    echo "阶段2：跳过（下游阶段已有完整 checkpoint）"
    STAGE2_CKPT=""
else
    echo "[阶段2配置] 输入模型=${STAGE1_CKPT}/model.pt; lr=0.008; epoch=15; 训练=body.72之后的backbone，PartialFC冻结，其他BN统计冻结"
    run_or_resume_stage \
        "阶段2：body.72 解冻（PartialFC 冻结）" \
        "$STAGE2_PREFIX" \
        "models.start_from=${STAGE1_CKPT}/model.pt" \
        "models.freeze=True" \
        "classifiers.freeze=True" \
        "data_augs=configs/gridsample_v2_numpy.yaml" \
        "pefts=configs/part_freeze.yaml" \
        "pefts.target_modules=body.72" \
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
fi

echo "[阶段2输出] checkpoint=${STAGE2_CKPT}"
echo "[阶段3配置] 输入模型=${STAGE2_CKPT}/model.pt; lr=0.006; epoch=5; 训练=PartialFC，backbone冻结"
run_or_resume_stage \
    "阶段3：PartialFC 重训练（backbone 冻结）" \
    "$STAGE3_PREFIX" \
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
    "$STAGE4_PREFIX" \
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
echo "TopoFR IR-200 四阶段训练完成"
echo "阶段1 checkpoint：${STAGE1_CKPT}"
echo "阶段2 checkpoint：${STAGE2_CKPT}"
echo "阶段3 checkpoint：${STAGE3_CKPT}"
echo "阶段4 checkpoint：$(latest_checkpoint "$STAGE4_PREFIX")"
echo "========================================="

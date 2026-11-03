#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 — TopoFR + IR200 变体 v2 (work_260922_topofr)
#
# 【v2 相对 four_stage_topofr_ir200.sh (v1) 的变更 — s2 冻结分类器】
#   v1 实跑结果: s2 ep0-2 的 34t@1e-6 健康 (90+), ep3 掉到 41, ep4 起归零 (~0.03)
#   且 s2 后半 + s3 + s4 全程未恢复; glint/IJBC/训练 loss 全程正常 (s4 loss 平滑
#   10.7→6.0)。即"超低 FAR 尾部判别力"在 s2 lr 峰值后被打崩, 中等指标无感。
#   根因判断 (证据: topofr 分支原配方 s2 冻结分类器在 0605 上全程健康; 崩点恰在
#   warmup 结束 lr 峰值 0.008 处): v1 沿用了主线 s2 分类器 0.1x lr 同步训练, 而
#   TopoFR 的 GUM 样本加权依赖分类器给出的 entropy 分布 — 分类器边动边加权形成
#   失稳反馈回路, 高 lr 下把 embedding 尾部打穿。
#   v2 修复 (方案 A, 回到 topofr 分支验证过的形态):
#     - s2: classifiers.freeze=True, 完全冻结分类器 (requires_grad=False, 不进
#       优化器), 分类器重对齐交给 s3; 不再传 classifier_lr_scale
#     - s4: 维持分类器 0.1x 同步 (joint 形态) — s4 lr 峰值低 10 倍 (0.0008),
#       且 topofr 分支的 s4joint 变体 (classifier_lr=0.0001) 验证过此形态
#   BN 说明: 无需也不应改动 — PFC 分类器无 BN 层; backbone 的 BN 由 pipeline
#   train() 内建处理 (models.freeze=True 时整网先 eval, 冻结段 BN 的 running
#   stats 不更新且无梯度, 仅解冻段 body.72+ 的 BN 恢复 train mode 适应新特征)。
#
# 【s1 复用】v1 的 s1 (分类器拟合, backbone 冻结) 产出健康 — s2 ep0 的 34t=90.6
#   即其质量证明。本脚本通过 S1_REUSE_DIR 软链复用 v1 s1 的 checkpoint, 不重训;
#   置空该变量则 v2 从头跑 s1。
#
# 【重要约束】(同 v1)
#   - bs 全程 128/卡, 勿随意调大: 拓扑损失只用本地 batch 且 O(N²) CPU 计算
#     (持久同调 N=128 约 30ms, N=1024 约 2289ms, 77 倍); GUM EM 也在 CPU。
#   - 拓扑项的 latent_norm 在 topology.py 里硬编码 .cuda(), 训练必须见 GPU。
#   - ir200 显存 ~1.7x ir101; OOM 优先降 bs (同时降拓扑耗时)。
#
# 模型:   iResNet-200 insightface (TopoFR, Glint360K 预训练)
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计 (超参同 v1, 仅 s2 分类器策略变更):
#   s1  分类器训练, backbone 全冻结                  step  lr=0.008, 5  epoch (复用 v1)
#   s2  body.72~97+output_layer 训练, 分类器冻结     cos  lr=0.008, 15 epoch
#   s3  分类器重对齐, backbone 全冻结                step  lr=0.006, 5  epoch
#   s4  全模型训练     + 分类器 0.1x lr              cos  lr=0.0008, 15 epoch
#
# 断点续传 (直接重跑本脚本即可, 幂等): 机制同 v1, 按 ${CAMPAIGN}_* 前缀扫描。
#
# 用法:  bash scripts/train/four_stage_topofr_ir200_v2.sh (建议放 nohup/screen 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="topofr_260922_v2"       # 战役前缀; v2 隔离 v1 崩掉的 s2/s3/s4 旧目录
NGPU=8                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface/cvlface/research/recognition/code/work_260922_topofr
SAVE_ROOT=/data1/dataset_260918/train_output
PRETRAINED=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200/Glint360K_R200_TopoFR_9784.pt
# 复用 v1 的健康 s1 (置空则 v2 从头训 s1)
S1_REUSE_DIR=/data1/dataset_260918/train_output/topofr_260922_s1_cls_09-30_1

# ----------------------------- 环境 (8卡机 qingsi) -----------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/root/anaconda3/envs/cvlface
export PATH=/root/anaconda3/envs/cvlface/bin:$PATH
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}   # scipy libstdc++ 冲突
export DECODE_BACKEND=turbojpeg

LOGDIR="${SAVE_ROOT}/shell_logs"
mkdir -p "${SAVE_ROOT}" "${LOGDIR}"
cd "${WORKDIR}"

TS=$(date +%m%d_%H%M%S)
MASTER_LOG="${LOGDIR}/four_stage_${CAMPAIGN}_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

# ----------------------------- 预检 -----------------------------
command -v fabric >/dev/null || { log "错误: fabric 不在 PATH (conda env cvlface?)"; exit 1; }
[ -f "${PRETRAINED}" ] || { log "错误: 预训练模型不存在 ${PRETRAINED}"; exit 1; }
for f in train_rec/train.rec train_rec/train.idx; do
    [ -f "/data1/dataset_260918/${f}" ] || { log "错误: 数据缺失 /data1/dataset_260918/${f}"; exit 1; }
done
NGPU_REAL=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU_REAL}" -lt "${NGPU}" ]; then
    log "错误: 仅可见 ${NGPU_REAL} 卡, 需要 ${NGPU}"; exit 1
fi
for d in val_34t val_enhance val_glint facerec_val/cplfw facerec_val/calfw \
         facerec_val/agedb_30 facerec_val/tinyface_aligned_pad_0.1; do
    [ -d "/data1/dataset_260918/${d}" ] || log "警告: 评估集目录缺失 /data1/dataset_260918/${d}"
done

# ----------------------- 断点检测 -----------------------
# 用法: latest_complete <prefix>  ->  stdout: "<epoch> <ckpt_dir>"; 无则 "-1 "
latest_complete() {
    local best=-1 bdir="" root ed n
    for root in "${SAVE_ROOT}"/${1}_*/checkpoints_every_epoch; do
        [ -d "${root}" ] || continue
        for ed in "${root}"/epoch:*; do
            [ -f "${ed}/pipeline.pt" ] || continue                       # rank0 最后写, 完整性标记
            [ -f "${ed}/model.pt" ] || continue
            [ -f "${ed}/classifier_rank$((NGPU-1)).pt" ] || continue
            n="${ed##*:}"
            if (( n > best )); then best=${n}; bdir="${ed}"; fi
        done
    done
    echo "${best} ${bdir}"
}

LAST_STAGE_DIR=""
# ----------------------- 阶段执行器 -----------------------
# 用法: run_stage <名> <prefix> <num_epoch> [hydra 覆盖参数...]
run_stage() {
    local name=$1 prefix=$2 num_epoch=$3; shift 3
    local out epoch dir rc try=0 prev_epoch
    while :; do
        out=$(latest_complete "${prefix}")
        epoch=${out%% *}; dir=${out#* }
        if (( epoch >= num_epoch - 1 )); then
            log "[${name}] 已完成 (epoch:${epoch}), ckpt: ${dir}"
            LAST_STAGE_DIR="${dir}"
            return 0
        fi
        local resume=()
        if (( epoch >= 0 )); then
            resume=("trainers.resume=${dir}")
            log "[${name}] 从 epoch:${epoch} 断点续训: ${dir}"
        else
            log "[${name}] 全新开始 (目标 epoch:$((num_epoch-1)))"
        fi
        # 注意: train_opt.py 内部自建 DDPStrategy, CLI 不能再传 --strategy (会冲突报错)
        fabric run --devices=${NGPU} --precision="bf16-mixed" \
            train_opt.py \
            trainers.prefix="${prefix}" trainers.num_gpu=${NGPU} \
            "$@" "${resume[@]}" \
            2>&1 | tee -a "${LOGDIR}/${name}_${TS}.log"
        rc=${PIPESTATUS[0]}
        if (( rc == 0 )); then
            out=$(latest_complete "${prefix}")
            epoch=${out%% *}; dir=${out#* }
            if (( epoch >= num_epoch - 1 )); then
                log "[${name}] 完成, ckpt: ${dir}"
                LAST_STAGE_DIR="${dir}"
                return 0
            fi
            log "[${name}] 正常退出但未达目标 epoch (最大 ${epoch}, 可能触发早停), 终止。"
            log "[${name}] 请查看 ${LOGDIR}/${name}_${TS}.log; 确认要继续可调大 num_epoch/早停耐心后重跑本脚本"
            exit 1
        fi
        # 非零退出: 有 epoch 进展则重置重试计数, 无进展则累计
        out=$(latest_complete "${prefix}"); epoch=${out%% *}
        if [ "${epoch}" -gt "${prev_epoch:--1}" ]; then
            try=0
        else
            try=$((try + 1))
        fi
        prev_epoch="${epoch}"
        if (( try >= MAX_RETRY )); then
            log "[${name}] 连续 ${try} 次无进展 (退出码 ${rc}), 终止。日志: ${LOGDIR}/${name}_${TS}.log"
            exit "${rc}"
        fi
        log "[${name}] 异常退出 (码 ${rc}, 重试 ${try}/${MAX_RETRY}), 30s 后自动续训"
        sleep 30
    done
}

# ----------------------- 公共参数 (四阶段一致) -----------------------
# epoch 末评估走外挂 TRT 单进程链路 (eval_all_trt_single), 指标键名一致。
# TopoFR 三件套: ir200 (insightface) + TopoFR-AdaFace 损失 + TopoFR pipeline。
# bs 固定 128: 拓扑损失 O(N²) 本地 batch 约束, 见文件头【重要约束】。
COMMON=(
    trainers.batch_size=128
    trainers.num_workers=8
    trainers.precision=bf16-mixed
    trainers.external_eval=True
    trainers.external_eval_backend=trt
    trainers.external_eval_precision=fp16
    models=iresnet_insightface/configs/v1_ir200.yaml
    dataset=configs/dataset_260922_train.yaml
    dataset.model_save_dir="${SAVE_ROOT}"
    classifiers=configs/partial_fc_sample10.yaml
    classifiers.sample_rate=0.40
    losses=configs/topofr_adaface.yaml
    pipelines=configs/train_model_cls_topofr.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== 四阶段 TopoFR+IR200 v2 (s2 冻结分类器) | campaign=${CAMPAIGN} | sr=0.40 bs=128 | ${NGPU} 卡 ===="

################################################################################
# s1: 分类器训练 (backbone 全冻结, step 调度, 5 epoch)
#     默认软链复用 v1 的健康 s1; 置空 S1_REUSE_DIR 则从头训
################################################################################
if [ -n "${S1_REUSE_DIR}" ] && [ -d "${S1_REUSE_DIR}/checkpoints_every_epoch" ] \
   && ! ls -d "${SAVE_ROOT}"/${CAMPAIGN}_s1_cls_* >/dev/null 2>&1; then
    reuse_link="${SAVE_ROOT}/${CAMPAIGN}_s1_cls_reuse$(date +_%m%d_%H%M%S)"
    ln -sfn "${S1_REUSE_DIR}" "${reuse_link}"
    log "[s1] 复用 v1 s1 checkpoint: ${S1_REUSE_DIR} -> ${reuse_link} (不重训)"
fi
run_stage s1 "${CAMPAIGN}_s1_cls" 5 \
    "${COMMON[@]}" \
    models.start_from="${PRETRAINED}" \
    models.freeze=True \
    data_augs=configs/basic_v2_numpy.yaml \
    pefts=configs/freeze.yaml \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=5 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0
S1_DIR="${LAST_STAGE_DIR}"

################################################################################
# s2: body.72~97+output_layer 训练, 分类器完全冻结 (cosine, 15 epoch)
#     v2 核心变更: classifiers.freeze=True — GUM 加权依赖分类器 entropy 分布,
#     冻结它以切断 v1 中失稳的反馈回路; 分类器重对齐交给 s3。
#     BN 无需额外处理: 冻结段 BN 已被 pipeline train() 保持 eval,
#     解冻段 body.72+ 的 BN 保持 train 适应新特征 (勿冻)。
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body72" 15 \
    "${COMMON[@]}" \
    models.start_from="${S1_DIR}/model.pt" \
    models.freeze=True \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.72 \
    pefts.classifier_ckpt_dir="${S1_DIR}" \
    classifiers.freeze=True \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine
S2_DIR="${LAST_STAGE_DIR}"

################################################################################
# s3: 分类器重对齐 (backbone 全冻结, step 调度, 5 epoch)
#     v2 里这一阶段变得关键: s2 冻结了分类器, 此处承接 s2 后的 backbone 重新拟合
################################################################################
run_stage s3 "${CAMPAIGN}_s3_cls" 5 \
    "${COMMON[@]}" \
    models.start_from="${S2_DIR}/model.pt" \
    models.freeze=True \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/freeze.yaml \
    pefts.classifier_ckpt_dir="${S2_DIR}" \
    optims=configs/step_sgd.yaml \
    optims.lr=0.006 optims.num_epoch=5 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0
S3_DIR="${LAST_STAGE_DIR}"

################################################################################
# s4: 全模型微调 + 分类器 0.1x lr 同步微调 (cosine, 15 epoch)
#     维持 joint 形态: s4 lr 峰值低 10 倍 (0.0008), topofr 分支的 s4joint 变体
#     (classifier_lr=0.0001) 验证过此形态; 若 v2 的 s4 仍出现 34t 崩塌,
#     下一版再改 classifiers.freeze=True (分支 s4 原配方) 对照
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 15 \
    "${COMMON[@]}" \
    models.start_from="${S3_DIR}/model.pt" \
    models.freeze=False \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/full.yaml \
    pefts.classifier_ckpt_dir="${S3_DIR}" \
    classifiers.freeze=False \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0008 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine \
    optims.classifier_lr_scale=0.1
S4_DIR="${LAST_STAGE_DIR}"

log "==== 四阶段 TopoFR+IR200 v2 训练全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name work_260922_topofr --name s4_final_v2 --timeout_minutes 90"

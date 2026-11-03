#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 — QGFace + SubCenter 变体 (work_260922_qgface_subcenter)
#
# 相对 four_stage_260922.sh 的差异 (技术移植自 qgface_subcenter 分支):
#   - 分类器: SubCenter PFC, 每类 K=3 个子中心, logit 取子中心内 amax
#     (classifiers=configs/partial_fc_subcenter_k3_sample40.yaml)
#   - 损失: AdaFace + quality_scale=0.2 质量门控 (losses=configs/qgface_adaface.yaml)
#   - 数据: QGFace 双视图 (data_augs=configs/qgface.yaml) — 底层 identity 增强,
#     ContrastiveViewDataset 生成 (query=降质视图, key=原图50%翻转), 每步图像
#     经 backbone 数 = 2 x batch_size
#   - pipeline: TrainQGFacePipeline (pipelines=configs/train_qgface.yaml) —
#     分类损失吃双视图, s2/s4 额外叠加 QGFace 质量引导对比损失 (负样本来自
#     proxy 实时队列, 子中心路由由原图视图特征决定); backbone 与 PFC 用独立
#     优化器/裁剪预算 (optims.classifier_lr 为分类器独立基准 lr)
#   - s1/s3: pipelines.contrast_weight=0.0 (只训分类器, 双视图仍进分类损失)
#   - s2/s4: pipelines.contrast_weight=1.0 (QG 对比损失开启)
#   - 分类器策略沿用主线: s2/s4 分类器与 backbone 同步 0.1x lr 训练
#
# 【显存注意】
#   - 双视图使每卡过 backbone 图像数 = 2 x batch_size。本脚本 bs 已相对基线减半
#     (s1-s3: 256 -> 每卡 512 图; s4: 64 -> 每卡 128 图)。s4 已按 subcenter 工作
#     区 OOM 教训 (提交 d66022d: K=3+sr0.4 下每卡 256 图激活超 24G) 降到 128 图,
#     lr 同步减半; 原 bs128 双视图 (256 图/卡) 与 OOM 场景等激活量。
#     若仍 OOM: 先减半 trainers.batch_size, 再降 classifiers.sample_rate。
#   - K=3 + sample_rate=0.40 使每卡激活中心行数 x3 (~12.4万行), logit 显存/耗时
#     高于 K=1 基线。
#   - QGFace 约束: pipelines.qgface.queue_size(8192) >= 2 x 全局 batch
#     (s1-s3: 256x8x2=4096 <= 8192 ✓; 若调大 bs 需同步调大 queue_size)。
#   - 分类器断点/衔接全程要求 K=3, 勿与 K=1 campaign 的 checkpoint 目录混用
#     (world_size 相同时形状校验会直接报错拦截)。
#
# 模型:   iResNet-101 (AdaFace, WebFace12M 预训练)
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计:
#   s1  分类器训练(K=3随机初始化), backbone 冻结, QG关闭  step  lr=0.008, 5  epoch
#   s2  body.36~48+output_layer + 分类器0.1x, QG开启        cos  lr=0.008, 15 epoch
#   s3  分类器再训练, backbone 冻结, QG关闭                 step  lr=0.006, 5  epoch
#   s4  全模型训练 + 分类器0.1x, QG开启                     cos  lr=0.0004, 15 epoch
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描该 prefix 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank7.pt 齐全, pipeline.pt 由 rank0 最后写入即完整性标记),
#     通过 trainers.resume 恢复 model/classifier/qgface_loss/双优化器/双调度器,
#     从 epoch+1 继续。注意每次重启 fabric 会新建 run 目录, 旧目录保留, 属预期。
#   - 阶段间: s2/s3/s4 自动读取上一阶段最后一个 epoch
#     (models.start_from=上阶段/model.pt 加载 backbone, pefts.classifier_ckpt_dir
#     加载分类器; 优化器/调度器/QG队列按新阶段重置)。
#   - 阶段完成判定: 存在完整 epoch:$(N-1)。早停等导致未达标时会中止并提示。
#
# 用法:  bash scripts/train/four_stage_qgface_subcenter.sh   (建议放 nohup/screen 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="qgface_subcenter_260922"   # 本次战役前缀; 换超参数开新战役时改它, 避免误续旧目录
NGPU=8                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface/cvlface/research/recognition/code/work_260922_qgface_subcenter
SAVE_ROOT=/data1/dataset_260918/train_output
PRETRAINED=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt

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
# epoch 末评估走外挂 TRT 单进程链路 (eval_all_trt_single: nvjpeg 解码 + TRT fp16
# + v7 匹配, 无 NCCL/fabric), 比进程内评估快数倍; 结果 JSON 回读后走同一套
# summary/best/早停 逻辑, 指标键名一致 (如 work_260922_34t/tpir_at_far_1e-10)。
# 注意: TRT fp16 与进程内 torch bf16 评估存在可预期的小口径差, 跨阶段对比时以此解释。
# QGFace 三件套: 双视图增强 + adaface 质量门控 + TrainQGFacePipeline (对比权重
# 逐阶段用 pipelines.contrast_weight 覆盖)。
COMMON=(
    trainers.num_workers=8
    trainers.precision=bf16-mixed
    trainers.external_eval=True
    trainers.external_eval_backend=trt
    trainers.external_eval_precision=fp16
    models=iresnet/configs/v1_ir101.yaml
    dataset=configs/dataset_260922_train.yaml
    dataset.model_save_dir="${SAVE_ROOT}"
    classifiers=configs/partial_fc_subcenter_k3_sample40.yaml
    losses=configs/qgface_adaface.yaml
    data_augs=configs/qgface.yaml
    pipelines=configs/train_qgface.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== 四阶段 QGFace+SubCenter 训练启动 | campaign=${CAMPAIGN} | K=3 sr=0.40 | ${NGPU} 卡 | 输出根: ${SAVE_ROOT} ===="

################################################################################
# s1: 分类器训练 (SubCenter K=3 随机初始化, backbone 全冻结, QG 关闭, step 调度, 5 epoch)
#     双视图仍进分类损失 (contrast_weight=0 只关对比项)
################################################################################
run_stage s1 "${CAMPAIGN}_s1_cls" 5 \
    trainers.batch_size=256 \
    "${COMMON[@]}" \
    models.start_from="${PRETRAINED}" \
    models.freeze=True \
    pipelines.contrast_weight=0.0 \
    pefts=configs/freeze.yaml \
    optims=configs/qgface_sgd.yaml \
    optims.lr=0.008 optims.classifier_lr=0.008 optims.num_epoch=5 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0 \
    optims.scheduler=step
S1_DIR="${LAST_STAGE_DIR}"

################################################################################
# s2: body.36~48+output_layer 训练 + 分类器 0.1x lr 同步微调 (cosine, 15 epoch)
#     QG 对比损失开启 (contrast_weight=1.0); 冻结段 BN 维持 eval, 解冻段 BN 随
#     train 更新 (train_opt.py 内建)
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body36" 15 \
    trainers.batch_size=256 \
    "${COMMON[@]}" \
    models.start_from="${S1_DIR}/model.pt" \
    models.freeze=True \
    pipelines.contrast_weight=1.0 \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.36 \
    pefts.classifier_ckpt_dir="${S1_DIR}" \
    classifiers.freeze=False \
    optims=configs/qgface_sgd.yaml \
    optims.lr=0.008 optims.classifier_lr=0.0008 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.max_grad_norm=5.0 \
    optims.scheduler=cosine
S2_DIR="${LAST_STAGE_DIR}"

################################################################################
# s3: 分类器再训练 (backbone 全冻结, QG 关闭, step 调度, 5 epoch), 承接 s2 后的 backbone
################################################################################
run_stage s3 "${CAMPAIGN}_s3_cls" 5 \
    trainers.batch_size=256 \
    "${COMMON[@]}" \
    models.start_from="${S2_DIR}/model.pt" \
    models.freeze=True \
    pipelines.contrast_weight=0.0 \
    pefts=configs/freeze.yaml \
    pefts.classifier_ckpt_dir="${S2_DIR}" \
    optims=configs/qgface_sgd.yaml \
    optims.lr=0.006 optims.classifier_lr=0.006 optims.num_epoch=5 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0 \
    optims.scheduler=step
S3_DIR="${LAST_STAGE_DIR}"

################################################################################
# s4: 全模型微调 + 分类器 0.1x lr 同步微调 (QG 开启, cosine, 15 epoch)
#     bs 64(双视图=每卡128图): 对齐 subcenter 工作区 s4 OOM 教训 (提交 d66022d,
#     K=3+sr0.4 下每卡 256 图激活超 24G; 原 bs128 双视图同为 256 图/卡, 同风险)
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 15 \
    trainers.batch_size=64 \
    "${COMMON[@]}" \
    models.start_from="${S3_DIR}/model.pt" \
    models.freeze=False \
    pipelines.contrast_weight=1.0 \
    pefts=configs/full.yaml \
    pefts.classifier_ckpt_dir="${S3_DIR}" \
    classifiers.freeze=False \
    optims=configs/qgface_sgd.yaml \
    optims.lr=0.0004 optims.classifier_lr=0.00004 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.max_grad_norm=5.0 \
    optims.scheduler=cosine
S4_DIR="${LAST_STAGE_DIR}"

log "==== 四阶段 QGFace+SubCenter 训练全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final --timeout_minutes 90"

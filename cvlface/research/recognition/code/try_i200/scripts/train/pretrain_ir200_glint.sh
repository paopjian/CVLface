#!/usr/bin/env bash
################################################################################
# IR-200 从头预训练 (try_i200, star 7x RTX4090)
#
# 模型:   iResNet-200 insightface 风格 (layer1[6]+layer2[26]+layer3[60]+layer4[6],
#         98 blocks, 118.8M 参数) — 随机初始化, 无预训练起点
# 数据:   /data1/dataset_260918/train_rec_glint (RecordIO, 300,194 类, 7,396,386 图,
#         2026-10-17 从全量 train_rec 按 train_v5.parquet source_group==glint 提取,
#         JPEG 字节与原库一致, 已过全量结构/解码/Reader 校验)
# 目标:   glint-only 域 IR200 预训练权重 (规模从主线 ir101 扩大), 供 IR200
#         对照实验: ① 后续四阶段微调 warm-start ② 与 TopoFR-R200-Glint360K 对比
# 评估:   仅开源基准 (cplfw/calfw/agedb_30/tinyface + IJBC_gt_aligned 模板级
#         TPIR@FPIR), 无内部 custom_verification4 — 预训练阶段不引入内部域指标
#
# 配方 (CVLface 上游 webface4m 从头训范式 × insightface glint360k-pfc 惯例):
#   - AdaFace 损失 (m=0.4 h=0.333 tα=0.01) + PartialFC sample_rate=0.1
#   - SGD momentum=0.9 wd=5e-4, lr=0.1 (有效 batch 224x7=1568), warmup 2 epoch
#     + cosine 退火, 共 26 epoch, max_grad_norm=5.0, filter_bias_and_bn
#   - bf16-mixed, turbojpeg 解码, 8 workers/卡
#   - 每 epoch 末外挂 TRT fp16 评估 (eval_all_trt_single 链, 与主线同口径)
#
# 【bs=224 的原因 (2026-10-17 GPU OOM 事故)】: 7 个 rank 进程都会在 GPU0 上各建
#   一个 ~384MB CUDA context (fabric/NCCL rendezvous 先于 set_device, 框架层行为),
#   GPU0 比 GPU1-6 少 ~2.3GB; bs=256 时 GPU0 余量 <20MB, epoch1 随机峰值 OOM。
#   bs 224 全卡腾出 ~2.6GB 激活内存 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#   防碎片, GPU0 余量恢复到 ~2GB。勿在未确认 GPU0 context 现象消失前改回 256。
#
# 时间预算 (ir101 bs512x7 实测 14,586 img/s 锚点, ir200 FLOPs x1.7 折算):
#   训练 ~15 min/epoch, 评估 ~5-8 min/epoch → 全程约 9-12 小时
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   扫描 ${SAVE_ROOT}/${CAMPAIGN}_*/checkpoints_every_epoch 取全局最大完整 epoch
#   (pipeline.pt + model.pt + classifier_rank6.pt 齐全), trainers.resume 恢复,
#   从 epoch+1 继续; 重启会新建 run 目录 (fabric 命名), 旧目录保留, 属预期。
#
# 用法:  bash scripts/train/pretrain_ir200_glint.sh     (建议放 tmux/nohup 里)
# 日志:  ${SAVE_ROOT}/shell_logs/pretrain_${CAMPAIGN}_<ts>.log
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="ir200_glint_pretrain"  # 战役前缀; 换配方开新战役时改它, 避免误续旧目录
NUM_EPOCH=26                     # 总 epoch (cosine 长度, 中途改会破坏调度曲线)
NGPU=7                           # GPU 数 (PFC 按 rank 分片, 中途不可改; GPU7 故障禁用)
MAX_RETRY=3                      # 连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface_rec/cvlface/research/recognition/code/try_i200
SAVE_ROOT=/data1/dataset_260918/train_output

# ----------------------------- 环境 (star 192.168.2.100) -----------------------------
# 本机 cvlface 环境在 miniconda3 下 (/root/anaconda3/envs/cvlface 是另一台 8 卡机
# 的路径, 不可混用); trainers/configs/default.yaml 的 external_eval_fabric_bin
# 已同步指向 miniconda (指错则每 epoch 外挂评估 FileNotFoundError)。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export CUDA_HOME=/root/miniconda3/envs/cvlface
export PATH=/root/miniconda3/envs/cvlface/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}   # scipy libstdc++ CXXABI 冲突
export DECODE_BACKEND=turbojpeg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 防 GPU0 碎片化 OOM (bs224 事故配套)

LOGDIR="${SAVE_ROOT}/shell_logs"
mkdir -p "${SAVE_ROOT}" "${LOGDIR}"
cd "${WORKDIR}"

TS=$(date +%m%d_%H%M%S)
MASTER_LOG="${LOGDIR}/pretrain_${CAMPAIGN}_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

# ----------------------------- 预检 -----------------------------
command -v fabric >/dev/null || { log "错误: fabric 不在 PATH (miniconda cvlface env?)"; exit 1; }
for f in train_rec_glint/train.rec train_rec_glint/train.idx train_rec_glint/train.tsv; do
    [ -f "/data1/dataset_260918/${f}" ] || { log "错误: 数据缺失 /data1/dataset_260918/${f}"; exit 1; }
done
NC=$(grep -o '"num_classes": [0-9]*' /data1/dataset_260918/train_rec_glint/meta.json | grep -o '[0-9]*')
[ "${NC}" = "300194" ] || { log "错误: train_rec_glint num_classes=${NC} != 300194 (数据被改动?)"; exit 1; }
for d in facerec_val/cplfw facerec_val/calfw facerec_val/agedb_30 \
         facerec_val/tinyface_aligned_pad_0.1 facerec_val/IJBC_gt_aligned; do
    [ -d "/data1/dataset_260918/${d}" ] || { log "错误: 评估集目录缺失 /data1/dataset_260918/${d}"; exit 1; }
done
NGPU_REAL=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU_REAL}" -lt "${NGPU}" ]; then
    log "错误: 仅可见 ${NGPU_REAL} 卡, 需要 ${NGPU}"; exit 1
fi
for pid in $(pgrep -f "train_opt.py" || true); do
    log "警告: 已有训练进程 ${pid} 在跑, 请先确认 GPU 空闲"; break
done

# ----------------------- 断点检测 -----------------------
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

# ----------------------- 训练循环 (带断点续跑) -----------------------
log "==== IR200 glint-only 预训练启动 | campaign=${CAMPAIGN} | ${NGPU} 卡 | ${NUM_EPOCH} epoch | 输出根: ${SAVE_ROOT} ===="

prev_epoch=-1
try=0
while :; do
    out=$(latest_complete "${CAMPAIGN}")
    epoch=${out%% *}; dir=${out#* }
    if (( epoch >= NUM_EPOCH - 1 )); then
        log "[pretrain] 已完成 (epoch:${epoch}), 最终 ckpt: ${dir}"
        log "[pretrain] 后续微调接入: models.start_from=${dir}/model.pt (分类器不需要带走)"
        break
    fi
    resume=()
    if (( epoch >= 0 )); then
        resume=("trainers.resume=${dir}")
        log "[pretrain] 从 epoch:${epoch} 断点续训: ${dir}"
    else
        log "[pretrain] 全新开始 (从头随机初始化, 目标 epoch:$((NUM_EPOCH-1)))"
    fi

    # 注意: train_opt.py 内部自建 DDPStrategy, CLI 不能再传 --strategy (会冲突报错)
    fabric run --devices=${NGPU} --precision="bf16-mixed" \
        train_opt.py \
        trainers.prefix="${CAMPAIGN}" trainers.num_gpu=${NGPU} \
        trainers.batch_size=224 \
        trainers.num_workers=8 \
        trainers.precision=bf16-mixed \
        trainers.external_eval=True \
        trainers.external_eval_backend=trt \
        trainers.external_eval_precision=fp16 \
        trainers.skip_final_eval=True \
        models=iresnet_insightface/configs/v1_ir200.yaml \
        models.freeze=False \
        dataset=configs/dataset_260922_train_glint.yaml \
        dataset.model_save_dir="${SAVE_ROOT}" \
        classifiers=configs/partial_fc_sample10.yaml \
        classifiers.sample_rate=0.1 \
        losses=configs/adaface.yaml \
        data_augs=configs/gridsample_v2_numpy.yaml \
        pefts=configs/full.yaml \
        pipelines=configs/train_model_cls.yaml \
        evaluations=configs/val_opensource_pretrain.yaml \
        optims=configs/step_sgd.yaml \
        optims.lr=0.1 optims.num_epoch=${NUM_EPOCH} optims.warmup_epoch=2 \
        optims.momentum=0.9 optims.weight_decay=0.0005 \
        optims.scheduler=cosine optims.classifier_lr_scale=1.0 \
        "optims.lr_milestones=[]" optims.max_grad_norm=5.0 \
        "${resume[@]}" \
        2>&1 | tee -a "${LOGDIR}/pretrain_${TS}.log"
    rc=${PIPESTATUS[0]}

    if (( rc == 0 )); then
        out=$(latest_complete "${CAMPAIGN}")
        epoch=${out%% *}; dir=${out#* }
        if (( epoch >= NUM_EPOCH - 1 )); then
            log "[pretrain] 完成, 最终 ckpt: ${dir}"
            log "[pretrain] 后续微调接入: models.start_from=${dir}/model.pt"
            break
        fi
        log "[pretrain] 正常退出但未达目标 epoch (最大 ${epoch}), 终止。日志: ${LOGDIR}/pretrain_${TS}.log"
        exit 1
    fi
    # 非零退出: 有 epoch 进展则重置重试计数, 无进展则累计
    out=$(latest_complete "${CAMPAIGN}"); epoch=${out%% *}
    if [ "${epoch}" -gt "${prev_epoch}" ]; then
        try=0
    else
        try=$((try + 1))
    fi
    prev_epoch=${epoch}
    if (( try >= MAX_RETRY )); then
        log "[pretrain] 连续 ${try} 次无进展 (退出码 ${rc}), 终止。日志: ${LOGDIR}/pretrain_${TS}.log"
        exit "${rc}"
    fi
    log "[pretrain] 异常退出 (码 ${rc}, 重试 ${try}/${MAX_RETRY}), 30s 后自动续训"
    sleep 30
done

log "==== 预训练全部完成 ===="

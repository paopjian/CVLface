#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 — SubCenter + IR-200 变体 (work_260922_ir200_subcenter)
#
# 【服务器环境 — 本脚本仅适配此机器, 换机器必改下面三处路径】
#   服务器:   qingsi (192.168.2.103, 8x RTX 4090, 数据在 /data1/dataset_260918)
#   cvlface:  /root/anaconda3/envs/cvlface (torch 2.12+cu126, python 3.12)
#   必改三处: ① 本节 export 的环境路径 ② trainers/configs/default.yaml 的
#             external_eval_fabric_bin ③ WORKDIR/PRETRAINED 等绝对路径
#
# 相对 four_stage_subcenter.sh 的差异 (IR200 化, 模式取自 topofr ir200 脚本):
#   - backbone: iResNet-200 insightface 风格 ([6,26,60,6]=98块, 118.8M 参数,
#     23.5 GFLOPs) — models/iresnet_insightface 包, 与 TreB1eN 风格权重不互通
#   - 预训练起点: 自产 IR200-glint 预训练权重 (try_i200, 2026-10-18 完成,
#     26 epoch 从头训: agedb 98.22 / tinyface r1 72.83 / IJBC@1e-5 94.77),
#     md5=10206f92610c1305db5a2bd12914dd78
#   - s2 解冻点: body.72 (= layer1.6+layer2.26+layer3 前 40 块之后, 与 ir101
#     的 body.36 同比例 ~73% 深度处; pefts/__init__.py 已从 topofr 带入
#     insightface ir100/ir200 的 body.K 映射)
#   - bs: s1/s3 冻结态保持 512; s2 降至 256 (ir200 解冻段激活内存 ~1.7x
#     ir101); s4 保持 128 (subcenter K=3 PFC 的既定选择)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (GPU0 碎片化 OOM 防护;
#     8 rank 各在 GPU0 建 ~384MB context, GPU0 比 GPU1-7 少 ~2.3GB)
#
# 模型:   iResNet-200 (insightface, IR200-glint 预训练 [自产])
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计 (超参数同 four_stage_subcenter):
#   s1  分类器训练(K=3随机初始化), backbone 冻结  step  lr=0.008, 5  epoch, bs 512
#   s2  body.72~97+output_layer + 分类器 0.1x lr  cos  lr=0.008, 15 epoch, bs 256
#   s3  分类器再训练, backbone 冻结               step  lr=0.006, 5  epoch, bs 512
#   s4  全模型训练 + 分类器 0.1x lr               cos  lr=0.0004, 15 epoch, bs 128
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描该 prefix 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank$((NGPU-1)).pt 齐全), trainers.resume 恢复。
#   - 阶段间: s2/s3/s4 自动读取上一阶段最后一个 epoch。
#   - 分类器断点/衔接均要求全程 K=3 (823875x3 分片)。
#
# 用法:  bash scripts/train/four_stage_ir200_subcenter.sh   (建议放 tmux/nohup 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="ir200_subcenter_260922"   # 战役前缀; 换超参数开新战役时改它
NGPU=8                         # GPU 数 (PFC 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface/cvlface/research/recognition/code/work_260922_ir200_subcenter
SAVE_ROOT=/data1/dataset_260918/train_output
PRETRAINED=/data2/zkj-data/ir200_glint_pretrain/model.pt

# ----------------------------- 环境 (qingsi 192.168.2.103, 8 卡) -----------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/root/anaconda3/envs/cvlface
export PATH=/root/anaconda3/envs/cvlface/bin:$PATH
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}   # scipy libstdc++ 冲突
export DECODE_BACKEND=turbojpeg
# 【勿开 expandable_segments】外挂评估的 v7 相似度 JIT 扩展(fused_he_pn_v1)与
# expandable_segments 的 VA 布局不兼容, 训练内评估必崩 CUDA illegal memory access
# (2026-10-18 排查: 手动无此变量评估成功/训练内继承此变量必崩/九月 subcenter 无此变量正常)。
# GPU0 碎片化风险由 bs 预留余量覆盖 (s2 bs256 实测每卡余 ~4GB)。

LOGDIR="${SAVE_ROOT}/shell_logs"
mkdir -p "${SAVE_ROOT}" "${LOGDIR}"
cd "${WORKDIR}"

TS=$(date +%m%d_%H%M%S)
MASTER_LOG="${LOGDIR}/four_stage_${CAMPAIGN}_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

# ----------------------------- 预检 -----------------------------
command -v fabric >/dev/null || { log "错误: fabric 不在 PATH (anaconda cvlface env?)"; exit 1; }
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
            log "[${name}] 正常退出但未达目标 epoch (最大 ${epoch}), 终止。"
            exit 1
        fi
        out=$(latest_complete "${prefix}"); epoch=${out%% *}
        if [ "${epoch}" -gt "${prev_epoch:--1}" ]; then
            try=0
        else
            try=$((try + 1))
        fi
        prev_epoch=${epoch}
        if (( try >= MAX_RETRY )); then
            log "[${name}] 连续 ${try} 次无进展 (退出码 ${rc}), 终止。日志: ${LOGDIR}/${name}_${TS}.log"
            exit "${rc}"
        fi
        log "[${name}] 异常退出 (码 ${rc}, 重试 ${try}/${MAX_RETRY}), 30s 后自动续训"
        sleep 30
    done
}

# ----------------------- 公共参数 (四阶段一致) -----------------------
COMMON=(
    trainers.num_workers=8
    trainers.precision=bf16-mixed
    trainers.external_eval=True
    trainers.external_eval_backend=trt
    trainers.external_eval_precision=fp16
    models=iresnet_insightface/configs/v1_ir200.yaml
    dataset=configs/dataset_260922_train.yaml
    dataset.model_save_dir="${SAVE_ROOT}"
    classifiers=configs/partial_fc_subcenter_k3_sample40.yaml
    losses=configs/adaface.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== 四阶段 IR200-SubCenter 训练启动 | campaign=${CAMPAIGN} | ${NGPU} 卡 | 预训练: ${PRETRAINED} ===="

################################################################################
# s1: 分类器训练 (backbone 全冻结, step 调度, 5 epoch, bs 512)
################################################################################
run_stage s1 "${CAMPAIGN}_s1_cls" 5 \
    trainers.batch_size=512 \
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
# s2: body.72~97+output_layer 训练 + 分类器 0.1x lr 同步微调 (cosine, 15 epoch, bs 256)
#     body.72 = ir200 的 ~73% 深度 (对应 ir101 的 body.36); pefts 映射已带入
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body72" 15 \
    trainers.batch_size=256 \
    "${COMMON[@]}" \
    models.start_from="${S1_DIR}/model.pt" \
    models.freeze=True \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.72 \
    pefts.classifier_ckpt_dir="${S1_DIR}" \
    classifiers.freeze=False \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine \
    optims.classifier_lr_scale=0.1
S2_DIR="${LAST_STAGE_DIR}"

################################################################################
# s3: 分类器再训练 (backbone 全冻结, step 调度, 5 epoch, bs 512)
################################################################################
run_stage s3 "${CAMPAIGN}_s3_cls" 5 \
    trainers.batch_size=512 \
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
# s4: 全模型微调 + 分类器 0.1x lr (cosine, 15 epoch, bs 128)
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 15 \
    trainers.batch_size=128 \
    "${COMMON[@]}" \
    models.start_from="${S3_DIR}/model.pt" \
    models.freeze=False \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/full.yaml \
    pefts.classifier_ckpt_dir="${S3_DIR}" \
    classifiers.freeze=False \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0004 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine \
    optims.classifier_lr_scale=0.1
S4_DIR="${LAST_STAGE_DIR}"

log "==== 四阶段全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final --timeout_minutes 90"

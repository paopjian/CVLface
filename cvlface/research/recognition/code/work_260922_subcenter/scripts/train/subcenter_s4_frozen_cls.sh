#!/usr/bin/env bash
################################################################################
# 单阶段微调训练 — SubCenter 变体, 仅重跑 s4 且分类器冻结 (work_260922_subcenter)
#
# 【服务器环境 — 本脚本仅适配此机器, 换机器必改下面三处路径】
#   服务器:   star (192.168.2.100, 7x RTX 4090, 数据在 /data1/dataset_260918)
#   cvlface:  /root/miniconda3/envs/cvlface (torch 2.12+cu126, python 3.12)
#   必改三处: ① 本节 export 的环境路径 ② trainers/configs/default.yaml 的
#             external_eval_fabric_bin (外挂评估的 python 由它推导, 指错则每
#             epoch 评估 FileNotFoundError, 见 2026-09-24 s2 全程 eval 失败事故)
#             ③ WORKDIR/SAVE_ROOT 等绝对路径
#   对照:     另一台 8 卡机的环境在 /root/anaconda3/envs/cvlface (原 work_260922
#             的 four_stage_260922.sh 用它), 两台机器路径不可混用。
#
# 相对 four_stage_subcenter_s2_frozen_cls.sh 的差异 (本脚本的全部改动):
#   - 只跑 s4, 不跑 s1~s3: backbone 与分类器均从既有 campaign
#     subcenter_fzcls_260922 的 s3 末完整 epoch 接力 (启动时扫描定位, 缺失即中止)。
#   - s4 分类器冻结不训练: classifiers.freeze=True 使分类器权重全程
#     requires_grad=False (classifiers/__init__.py), 不进优化器 (optims.py 对
#     无可训练参数的分类器不加参数组), 故相对原 s4 去掉
#     optims.classifier_lr_scale=0.1 (冻结后无分类器参数组, 该项失效)。
#     s3 末的分类器权重作为固定度量锚仅参与前向 AdaFace margin loss
#     (batch_mean/std 自适应仍随前向更新); 本阶段只训练 backbone 全模型。
#     动机: 原方案 s4 分类器 0.1x lr 与 backbone 同训, 两方更新互相拉扯;
#     推理只用 backbone embedding, 冻结分类器让 s4 微调完全围绕固定度量锚进行。
#   - campaign 前缀独立为 subcenter_fzcls_s4fz_260922: 断点扫描按前缀 glob 匹配,
#     与旧 s4 (subcenter_fzcls_260922_s4_full, 分类器可训版) 的目录互不可见,
#     旧 s4 的产出不会被误续; 两套可并行/先后跑, 互不误续。
#   - 显存: 冻结分类器省去其权重梯度+动量, 每卡约省 1.4G (~72.4万中心行/卡
#     x 512 维 x fp32 x2); bs 128/lr 0.0004 维持原 s4 保守档不动, 如需提档
#     可试 bs 256 + lr 0.0008 (2026-09-24 原版 s4 在该档 OOM, 冻结后有富余
#     但未实测, 谨慎起见默认仍 128)。
#
# 模型:   iResNet-101 (AdaFace, WebFace12M 预训练)
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计 (超参数同 four_stage_subcenter_s2_frozen_cls 的 s4, 除分类器冻结外):
#   s4  全模型训练, 分类器冻结                         cos  lr=0.0004, 15 epoch (bs 128)
#   承接: backbone <- s3末/model.pt, 分类器 <- s3末 classifier 分片 (K=3)
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描本 campaign 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank$((NGPU-1)).pt 齐全, pipeline.pt 由 rank0 最后写入即
#     完整性标记), 通过 trainers.resume 恢复 model/classifier/optimizer/scheduler,
#     从 epoch+1 继续。注意每次重启 fabric 会新建 run 目录 (日期+trial 命名),
#     旧目录保留, 属预期。
#   - 分类器断点: 冻结分类器不产生新权重, 各 epoch checkpoint 里的
#     classifier_rank*.pt 均等于 s3 末权重; resume 时按同形状校验加载, 无副作用。
#   - 注意: 本脚本的分类器断点/衔接均要求 K=3 (823875x3 分片) 且承接自
#     subcenter_fzcls_260922_s3_cls, 勿与 K=1 campaign 的 checkpoint 目录混用
#     (world_size 相同时形状校验会直接报错拦截)。
#
# 用法:  bash scripts/train/subcenter_s4_frozen_cls.sh   (建议放 nohup/screen 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="subcenter_fzcls_s4fz_260922"  # 本次战役前缀; 与旧 s4 (分类器可训版) 隔离 (见文件头)
S3_PREFIX="subcenter_fzcls_260922_s3_cls"  # 承接来源: s2 冻结分类器战役的 s3 阶段前缀
NGPU=7                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_260922_subcenter
SAVE_ROOT=/data1/dataset_260918/train_output

# ----------------------------- 环境 (star 192.168.2.100) -----------------------------
# 注: 本机 cvlface 环境在 miniconda3 下; /root/anaconda3/envs/cvlface 是另一台
# 8 卡机的路径, 不可混用 (见文件头【服务器环境】)。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export CUDA_HOME=/root/miniconda3/envs/cvlface
export PATH=/root/miniconda3/envs/cvlface/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}   # scipy libstdc++ 冲突
export DECODE_BACKEND=turbojpeg

LOGDIR="${SAVE_ROOT}/shell_logs"
mkdir -p "${SAVE_ROOT}" "${LOGDIR}"
cd "${WORKDIR}"

TS=$(date +%m%d_%H%M%S)
MASTER_LOG="${LOGDIR}/s4_frozen_cls_${CAMPAIGN}_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

# ----------------------------- 预检 -----------------------------
command -v fabric >/dev/null || { log "错误: fabric 不在 PATH (conda env cvlface?)"; exit 1; }
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
            if (( n > best )); then best="${n}"; bdir="${ed}"; fi
        done
    done
    echo "${best} ${bdir}"
}

# ----------------------- 承接来源检查 (s3 末 epoch) -----------------------
out=$(latest_complete "${S3_PREFIX}")
S3_EPOCH=${out%% *}; S3_DIR=${out#* }
if (( S3_EPOCH < 0 )); then
    log "错误: 未找到 ${S3_PREFIX} 的完整 checkpoint, 无法承接 s4。"
    log "请先跑完 four_stage_subcenter_s2_frozen_cls.sh 的 s3, 或修正 S3_PREFIX。"
    exit 1
fi
if (( S3_EPOCH < 4 )); then
    log "错误: ${S3_PREFIX} 最完整 epoch 为 ${S3_EPOCH} (< 目标 5 epoch 的末 epoch 4), s3 疑似未跑完。"
    log "请先用 four_stage_subcenter_s2_frozen_cls.sh 补完 s3 再跑本脚本 (避免从半成品分类器接续)。"
    exit 1
fi
log "承接来源确认: ${S3_PREFIX} epoch:${S3_EPOCH} -> ${S3_DIR}"

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

# ----------------------- 公共参数 (与四阶段脚本一致) -----------------------
# epoch 末评估走外挂 TRT 单进程链路 (eval_all_trt_single: nvjpeg 解码 + TRT fp16
# + v7 匹配, 无 NCCL/fabric), 比进程内评估快数倍; 结果 JSON 回读后走同一套
# summary/best/早停 逻辑, 指标键名一致 (如 subcenter_fzcls_s4fz_260922_34t/tpir_at_far_1e-10)。
# 注意: TRT fp16 与进程内 torch bf16 评估存在可预期的小口径差, 跨阶段对比时以此解释。
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
    losses=configs/adaface.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== SubCenter s4 重训 (分类器冻结, 仅训 backbone) 启动 | campaign=${CAMPAIGN} | K=3 sr=0.40 | 承接 ${S3_DIR} | ${NGPU} 卡 | 输出根: ${SAVE_ROOT} ===="

################################################################################
# s4: 全模型微调 (backbone 全解训), 分类器冻结 (cosine, 15 epoch)
#     与原 s4 (four_stage_subcenter_s2_frozen_cls.sh) 的差异:
#     - classifiers.freeze=True: 分类器由 pefts.classifier_ckpt_dir 从 s3 加载后
#       全程 requires_grad=False, 不进优化器 (无分类器参数组, 故去掉
#       classifier_lr_scale=0.1); 仅作固定度量锚参与前向 AdaFace margin loss。
#     - backbone: models.freeze=False + pefts=configs/full.yaml 全解训,
#       weights 由 models.start_from 从 s3 末 model.pt 加载。
#     显存: 冻结分类器每卡省约 1.4G (无分类器权重梯度/动量); bs 128 维持原
#     s4 保守档 (2026-09-24 bs 256 OOM 事故, 见文件头显存注)。
#     冻结段 BN 维持 eval 仅分类器内; backbone BN 随 train 更新 (train_opt.py 内建)。
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 15 \
    trainers.batch_size=128 \
    "${COMMON[@]}" \
    models.start_from="${S3_DIR}/model.pt" \
    models.freeze=False \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/full.yaml \
    pefts.classifier_ckpt_dir="${S3_DIR}" \
    classifiers.freeze=True \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0004 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine
S4_DIR="${LAST_STAGE_DIR}"

log "==== SubCenter s4 重训 (分类器冻结) 训练完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final_s4fz --timeout_minutes 90"

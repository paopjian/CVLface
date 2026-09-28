#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 (work_260922, 8x RTX4090)
#
# 模型:   iResNet-101 (AdaFace, WebFace12M 预训练)
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计 (超参数同 work_0605/流程.txt, 本轮改动: s2/s4 分类器同步小步训练):
#   s1  分类器训练, backbone 全冻结                  step  lr=0.008, 5  epoch
#   s2  body.36~48+output_layer 训练 + 分类器 0.1x lr  cos  lr=0.008, 15 epoch
#   s3  分类器训练, backbone 全冻结                  step  lr=0.006, 5  epoch
#   s4  全模型训练     + 分类器 0.1x lr              cos  lr=0.0008, 15 epoch
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描该 prefix 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank7.pt 齐全, pipeline.pt 由 rank0 最后写入即完整性标记),
#     通过 trainers.resume 恢复 model/classifier/optimizer/scheduler, 从 epoch+1 继续。
#     注意每次重启 fabric 会新建 run 目录 (日期+trial 命名), 旧目录保留, 属预期。
#   - 阶段间: s2/s3/s4 自动读取上一阶段最后一个 epoch
#     (models.start_from=上阶段/model.pt 加载 backbone, pefts.classifier_ckpt_dir
#     加载分类器; 优化器/调度器按新阶段重置)。
#   - 阶段完成判定: 存在完整 epoch:$(N-1)。早停等导致未达标时会中止并提示。
#
# 用法:  bash scripts/train/four_stage_260922.sh          (建议放 nohup/screen 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="adaface_260922"          # 本次战役前缀; 换超参数开新战役时改它, 避免误续旧目录
NGPU=8                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface/cvlface/research/recognition/code/work_260922
SAVE_ROOT=/data1/dataset_260918/train_output
PRETRAINED=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt

# ----------------------------- 环境 -----------------------------
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
# epoch 末评估走外挂 TRT 单进程链路 (eval_all_trt_single: nvjpeg 解码 + TRT fp16
# + v7 匹配, 无 NCCL/fabric), 比进程内评估快数倍; 结果 JSON 回读后走同一套
# summary/best/早停 逻辑, 指标键名一致 (如 work_260922_34t/tpir_at_far_1e-10)。
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
    classifiers=configs/partial_fc_sample10.yaml
    classifiers.sample_rate=0.40
    losses=configs/adaface.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== 四阶段训练启动 | campaign=${CAMPAIGN} | ${NGPU} 卡 | 输出根: ${SAVE_ROOT} ===="

################################################################################
# s1: 分类器训练 (backbone 全冻结, step 调度, 5 epoch)
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
# s2: body.36~48+output_layer 训练 + 分类器 0.1x lr 同步微调 (cosine, 15 epoch)
#     冻结段 BN 维持 eval, 解冻段 BN 随 train 更新 (train_opt.py 内建)
# wandb_run_id 钉在 09-22 首发的原 run (2c18pov1), 使 03-58 重启后的 epoch 合并
# 回同一曲线; s2 完成后此参数不再被使用, 留着无害。其余阶段 id 自动生成/续连。
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body36" 15 \
    trainers.batch_size=512 \
    "${COMMON[@]}" \
    trainers.wandb_run_id=2c18pov1 \
    models.start_from="${S1_DIR}/model.pt" \
    models.freeze=True \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.36 \
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
# s3: 分类器再训练 (backbone 全冻结, step 调度, 5 epoch), 承接 s2 后的 backbone
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
# s4: 全模型微调 + 分类器 0.1x lr 同步微调 (cosine, 15 epoch, bs 256)
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 15 \
    trainers.batch_size=256 \
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

log "==== 四阶段全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final --timeout_minutes 90"

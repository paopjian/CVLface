#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 — SubCenter + OHEM 变体 (work_260922_subcenter_ohem)
#
# 【服务器环境 — 本脚本仅适配此机器, 换机器必改下面三处路径】
#   服务器:   star (192.168.2.100, 7x RTX 4090, 数据在 /data1/dataset_260918)
#   cvlface:  /root/miniconda3/envs/cvlface (torch 2.12+cu126, python 3.12)
#   必改三处: ① 本节 export 的环境路径 ② trainers/configs/default.yaml 的
#             external_eval_fabric_bin (外挂评估的 python 由它推导, 指错则每
#             epoch 评估 FileNotFoundError, 见 2026-09-24 s2 全程 eval 失败事故)
#             ③ WORKDIR/PRETRAINED 等绝对路径
#   对照:     另一台 8 卡机的环境在 /root/anaconda3/envs/cvlface (原 work_260922
#             的 four_stage_260922.sh 用它), 两台机器路径不可混用。
#
# 相对 four_stage_subcenter.sh (本区前身) 的差异:
#   - 在 DistCrossEntropy 里加入 OHEM (online hard example mining, 分类版):
#     每步按 per-sample loss 取 top-k 困难样本聚合, k = ceil(ratio*B),
#     实现在 classifiers/partial_fc/partial_fc.py, 详见 docs/ohem_research.md。
#   - 默认 ohem_ratio=0.7 (前 1000 step 从 1.0 线性退火), 全四阶段生效;
#     临时调参无需改 yaml: run_stage 里追加 classifiers.ohem_ratio=0.8 即可。
#   - 日志新增 train/ohem_loss_all (全批均值, 可与基线对比)、train/ohem_loss_kept
#     (选中困难样本均值)、train/ohem_kept|ohem_batch|ohem_ratio_eff。
#   - SubCenter 本身不变: K=3, sample_rate=0.40
#     (classifiers=configs/partial_fc_subcenter_k3_sample40_ohem.yaml)
#   - 注意: 开 OHEM 后 train/loss 只聚合困难样本, 数值天然高于 subcenter 基线,
#     跨 run 对比请看 train/ohem_loss_all 与评估指标, 不要直接比 train/loss。
#
# 模型:   iResNet-101 (基础 = s4_0922_sub, 即 subcenter 基线战役 s4 产物, 见下)
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计 (s2/s4 轮次拉长 — 训到无收益策略, 领导 2026-10-28 指示):
#   s1  分类器训练, backbone 全冻结                  step  lr=0.008, 5  epoch (不变)
#   s2  body.36~48+output_layer 训练 + 分类器 0.1x lr  cos  lr=0.008, 30 epoch 上限
#   s3  分类器训练, backbone 全冻结                  step  lr=0.006, 5  epoch (不变)
#   s4  全模型训练     + 分类器 0.1x lr              cos  lr=0.0004, 30 epoch 上限 (bs 128, 见 s4 注)
# 轮次策略: s2/s4 上限 30, 验证指标连续 10 epoch 无改进即早停 (train_opt.py 内建
#   patience=10), 早停时已练 >= 15 (EARLYSTOP_ACCEPT_MIN, 不低于 subcenter 基线
#   训练量) 则视为阶段完成并衔接下一阶段。s1/s3 不拉长: 分类器-only 阶段 backbone
#   冻结、验证走 embedding 不变, 指标无信号可早停, 拉长轮次无法证伪。
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描该 prefix 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank$((NGPU-1)).pt 齐全, pipeline.pt 由 rank0 最后写入即完整性标记),
#     通过 trainers.resume 恢复 model/classifier/optimizer/scheduler, 从 epoch+1 继续。
#     注意每次重启 fabric 会新建 run 目录 (日期+trial 命名), 旧目录保留, 属预期。
#   - 阶段间: s2/s3/s4 自动读取上一阶段最后一个 epoch
#     (models.start_from=上阶段/model.pt 加载 backbone, pefts.classifier_ckpt_dir
#     加载分类器; 优化器/调度器按新阶段重置)。OHEM warmup 计数器存于分类器
#     checkpoint (partial_fc.dist_cross_entropy.ohem_step), 跨阶段继续累计。
#   - 阶段完成判定: 存在完整 epoch:$(N-1)。早停等导致未达标时会中止并提示。
#   - 分类器断点/衔接均要求全程 K=3 (823875x3 分片), 勿与 K=1 campaign 的
#     checkpoint 目录混用 (world_size 相同时形状校验会直接报错拦截)。
#   - 与 subcenter 基线 campaign 的 checkpoint 可互载 (weight 形状一致, OHEM
#     无新增 weight), 但 ohem_step 计数会随之迁移, 注意 warmup 语义。
#
# 用法:  bash scripts/train/four_stage_subcenter_ohem.sh       (建议放 nohup/screen 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="subcenter_ohem_260922"      # 本次战役前缀; 换超参数开新战役时改它, 避免误续旧目录
NGPU=7                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
# 早停被接受为"阶段完成"的最少 epoch 数 (= subcenter 基线 s2/s4 轮次)。
# 训练策略 (领导定): 加 OHEM + 拉长轮次, 只要不明显掉点就训到完全没收益 ——
# 即验证指标连续 patience=10 epoch 无改进时早停收工, 但训练量不得低于基线。
EARLYSTOP_ACCEPT_MIN=15
# done 看门狗宽限期 (分钟): train_opt.py 打印 done 后超过此时长进程仍未退出,
# 视为 NCCL teardown 挂死, 杀进程树按正常结束处理
DONE_GRACE_MIN=10
WORKDIR=/root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_260922_subcenter_ohem
SAVE_ROOT=/data1/dataset_260918/train_output
# 基础模型 (2026-10-28 领导指定): subcenter 基线战役的 s4 最终产物
# (subcenter_fzcls_260922_s4_full, K=3/sr0.4/adaface, ir101 512d) —— 本战役是
# 在已训完的 subcenter 基线上叠 OHEM 做第二轮增强微调; s1 仍随机初始化 K=3
# 分类器、在该冻结 backbone 上重新收敛 (s1 的职责)。
PRETRAINED=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/s4_0922_sub/model.pt

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
            if (( n > best )); then best="${n}"; bdir="${ed}"; fi
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
        # 早停完成标记: 整脚本重跑时不再把已早停的阶段续训到上限 (保持幂等)
        if (( epoch >= 0 )) && [ -f "${dir}/earlystop.done" ]; then
            log "[${name}] 已早停完成 (epoch:${epoch}, 标记存在), ckpt: ${dir}"
            LAST_STAGE_DIR="${dir}"
            return 0
        fi
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
        # done 看门狗: train_opt.py 打印 done 后仅剩 fabric/NCCL 退出清理, 偶发
        # destroy_process_group 跨 rank 挂死 (2026-10-28 s2 epoch17 早停后实测,
        # rank0 卡 teardown, GPU1-6 NCCL 自旋 100%)。done 出现后 DONE_GRACE_MIN
        # 分钟进程仍未退出则杀树按正常结束处理 —— checkpoint 在 done 前已全部落盘。
        local fab_log="${LOGDIR}/${name}_${TS}.log"
        fabric run --devices=${NGPU} --precision="bf16-mixed" \
            train_opt.py \
            trainers.prefix="${prefix}" trainers.num_gpu=${NGPU} \
            "$@" "${resume[@]}" \
            2>&1 | tee -a "${fab_log}" &
        local pipe_pid=$! done_min=0
        rc=""
        while kill -0 "${pipe_pid}" 2>/dev/null; do
            sleep 60
            if tail -n 5 "${fab_log}" | grep -q "^done$"; then
                done_min=$((done_min + 1))
                if (( done_min >= DONE_GRACE_MIN )); then
                    log "[${name}] 已打印 done 但 ${DONE_GRACE_MIN} 分钟未退出 (NCCL teardown 挂起), 清理进程树按正常结束处理"
                    pkill -TERM -f "train_opt.py trainers.prefix=${prefix}" 2>/dev/null
                    sleep 20
                    pkill -KILL -f "train_opt.py trainers.prefix=${prefix}" 2>/dev/null
                    wait "${pipe_pid}" 2>/dev/null
                    rc=0
                    break
                fi
            else
                done_min=0
            fi
        done
        if [[ -z "${rc}" ]]; then
            wait "${pipe_pid}"; rc=$?
        fi
        if (( rc == 0 )); then
            out=$(latest_complete "${prefix}")
            epoch=${out%% *}; dir=${out#* }
            if (( epoch >= num_epoch - 1 )); then
                log "[${name}] 完成, ckpt: ${dir}"
                LAST_STAGE_DIR="${dir}"
                return 0
            fi
            # 早停即完成: "训到无收益" (验证指标连续 patience=10 个 epoch 无改进) 是
            # 本战役的预期终止方式, 不再当异常中止。但至少要练满 EARLYSTOP_ACCEPT_MIN
            # (= subcenter 基线同阶段轮次) 才接受, 保证 A/B 对比训练量不低于基线。
            if (( epoch >= EARLYSTOP_ACCEPT_MIN )); then
                touch "${dir}/earlystop.done"
                log "[${name}] 早停完成 (验证无收益, 已练 ${epoch} epoch >= 下限 ${EARLYSTOP_ACCEPT_MIN}), ckpt: ${dir}"
                LAST_STAGE_DIR="${dir}"
                return 0
            fi
            log "[${name}] 正常退出但未达目标 epoch (最大 ${epoch} < 下限 ${EARLYSTOP_ACCEPT_MIN}), 判定为异常, 终止。"
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
# summary/best/早停 逻辑, 指标键名一致 (如 subcenter_260922_34t/tpir_at_far_1e-10)。
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
    classifiers=configs/partial_fc_subcenter_k3_sample40_ohem.yaml
    losses=configs/adaface.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== 四阶段 SubCenter+OHEM 训练启动 | campaign=${CAMPAIGN} | K=3 sr=0.40 ohem=0.7(warmup 1k) | ${NGPU} 卡 | 输出根: ${SAVE_ROOT} ===="

################################################################################
# s1: 分类器训练 (SubCenter K=3 随机初始化, backbone 全冻结, step 调度, 5 epoch)
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
# s2: body.36~48+output_layer 训练 + 分类器 0.1x lr 同步微调 (cosine, 30 epoch 上限)
#     冻结段 BN 维持 eval, 解冻段 BN 随 train 更新 (train_opt.py 内建)
#     15 -> 30: 拉长训练轮次; 验证指标 10 epoch 无改进则早停 (>=15 才接受),
#     "不掉点就训到没收益"。早停耐心/监控指标在 train_opt.py 内 (patience=10)。
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body36" 30 \
    trainers.batch_size=512 \
    "${COMMON[@]}" \
    models.start_from="${S1_DIR}/model.pt" \
    models.freeze=True \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.36 \
    pefts.classifier_ckpt_dir="${S1_DIR}" \
    classifiers.freeze=False \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=30 optims.warmup_epoch=2 \
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
# s4: 全模型微调 + 分类器 0.1x lr 同步微调 (cosine, 2026-11-01 领导指示改为训到
#     epoch 15 为止: num_epoch=16 即 epoch 0..15; 从 epoch:10 断点续训 5 个 epoch)
# 2026-09-24: bs 256/lr 0.0008 在 epoch0 第一个 backward 即 OOM (K=3+sr0.4 使
# 激活中心行 ~14.1万/卡, 全模型 backward 激活+logits fp32 临时超 24G)。
# 改 bs 128 + lr 减半 0.0004 (线性缩放规则); 若仍 OOM 下一档:
# 叠加 classifiers.sample_rate=0.2 和/或 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 16 \
    trainers.batch_size=128 \
    "${COMMON[@]}" \
    models.start_from="${S3_DIR}/model.pt" \
    models.freeze=False \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/full.yaml \
    pefts.classifier_ckpt_dir="${S3_DIR}" \
    classifiers.freeze=False \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0004 optims.num_epoch=16 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine \
    optims.classifier_lr_scale=0.1
S4_DIR="${LAST_STAGE_DIR}"

log "==== 四阶段 SubCenter 训练全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final --timeout_minutes 90"

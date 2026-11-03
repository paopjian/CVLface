#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 — TopoFR + IR200 变体 v2, s4 冻结分类器 (work_260922_topofr)
#
# 相对 four_stage_topofr_ir200.sh (v1) 的差异 (本脚本的全部改动):
#   - s4 分类器冻结不训练: classifiers.freeze=True 使分类器权重全程
#     requires_grad=False (classifiers/__init__.py), 不进优化器 (optims.py
#     has_trainable_params 门控, 无分类器参数组), 仅作 s3 训好的固定度量锚
#     参与前向 AdaFace margin loss; 相应去掉 optims.classifier_lr_scale
#     (冻结后该项失效)。动机: backbone 向固定分类器对齐, 避免原方案 (0.1x lr)
#     中两方更新互相拉扯造成的度量空间漂移 (同 work_260922_subcenter 的
#     s2 冻结分类器变体的做法, 此处用于最后一阶段 s4)。
#   - campaign 前缀独立为 topofr_fzcls_260922: 断点扫描按前缀 glob 匹配,
#     与 topofr_260922 (v1) 各阶段目录互不可见, 两套 campaign 可并行/先后跑,
#     互不误续。
#   - PRETRAINED 指向真实权重位置 /root/zhaokj/work_eval/models/
#     (v1 脚本里的 pretrained_models/recognition/topofr200/ 路径未实际放置)。
#   - s1/s2/s3 与 v1 完全一致; s4 其余超参与 v1 相同 (冻结分类器后省去分类器
#     梯度/动量显存, bs 128 维持不变 — 上限由拓扑损失 O(N²) 约束而非显存)。
#
# 相对 four_stage_260922.sh 的差异 (技术移植自 topofr 分支, 基于 work_0605):
#   - backbone: iResNet-200 (insightface 风格, layer1[6]+layer2[26]+layer3[60]+
#     layer4[6] = 98 blocks, 118.8M 参数; models/iresnet_insightface 包)
#     预训练: TopoFR 官方 Glint360K R200 权重 (注意 insightface 风格, 与
#     TreB1eN 风格 models/iresnet 的 200 层 [3,24,36,3] 权重不互通)
#   - 损失: TopoFRLoss 包装 AdaFace (losses=configs/topofr_adaface.yaml) —
#     持久同调拓扑损失 (输入/特征两空间 0 维 persistence pairs 对齐) +
#     GUM 高斯-均匀混合逐样本加权, 挂在 PartialFC 的 margin_softmax 位置
#   - pipeline: TrainModelClsTopoFRPipeline (仅比 TrainModelClsPipeline 多传
#     input_images 给分类器, 单优化器, 训练循环零侵入)
#   - s2 解冻点: ir200 用 body.72 (= layer1.6 + layer2.26 + layer3 前 40 块,
#     与 ir101 body.36 同比例 ~73% 深度处), pefts 已带 insightface ir200 映射
#
# 【重要约束】
#   - bs 全程 128/卡, 勿随意调大: 拓扑损失只用本地 batch 且 O(N²) CPU 计算
#     (持久同调 N=128 约 30ms, N=1024 约 2289ms, 77 倍); bs 翻倍则拓扑项时间
#     翻两番, 会吃掉大部分步时间。GUM EM 也在 CPU, 两者合计约 +15~20% 步时间。
#   - 拓扑项的 latent_norm 在 topology.py 里硬编码 .cuda(), 训练必须见 GPU。
#   - ir200 显存约为 ir101 的 ~1.7x, 128 bs 下 24GB 卡富余; 若 OOM 优先降 bs
#     (该方向同时降低拓扑耗时) 而不是动 sample_rate。
#
# 模型:   iResNet-200 insightface (TopoFR, Glint360K 预训练)
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计 (超参同 four_stage_260922 主线, 分类器 s2 同步 0.1x lr, s4 冻结):
#   s1  分类器训练, backbone 全冻结                  step  lr=0.008, 5  epoch
#   s2  body.72~97+output_layer 训练 + 分类器 0.1x lr  cos  lr=0.008, 15 epoch
#   s3  分类器训练, backbone 全冻结                  step  lr=0.006, 5  epoch
#   s4  全模型训练, 分类器冻结                       cos  lr=0.0008, 15 epoch
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描该 prefix 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank7.pt 齐全, pipeline.pt 由 rank0 最后写入即完整性标记),
#     通过 trainers.resume 恢复 model/classifier/optimizer/scheduler, 从 epoch+1 继续。
#     注意每次重启 fabric 会新建 run 目录 (日期+trial 命名), 旧目录保留, 属预期。
#   - 阶段间: s2/s3/s4 自动读取上一阶段最后一个 epoch
#     (models.start_from=上阶段/model.pt 加载 backbone, pefts.classifier_ckpt_dir
#     加载分类器; 优化器/调度器按新阶段重置)。
#     注: s4 分类器冻结, 其各 epoch checkpoint 里的 classifier_rank*.pt 与
#     s3 末 epoch 权重相同。
#   - 阶段完成判定: 存在完整 epoch:$(N-1)。早停等导致未达标时会中止并提示。
#
# 用法:  bash scripts/train/four_stage_topofr_ir200_v2_s4_frozen_cls.sh  (建议放 nohup/screen 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="topofr_fzcls_260922"   # 本次战役前缀; 与 v1 (topofr_260922) 隔离 (见文件头)
NGPU=8                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_260922_topofr
SAVE_ROOT=/data1/dataset_260918/train_output
# TopoFR 官方 R200 (Glint360K) 预训练, insightface 风格, 与 v1_ir200.yaml 匹配
PRETRAINED=/root/zhaokj/work_eval/models/Glint360K_R200_TopoFR_9784.pt

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
# summary/best/早停 逻辑, 指标键名一致 (如 topofr_fzcls_260922_34t/tpir_at_far_1e-10)。
# 注意: TRT fp16 与进程内 torch bf16 评估存在可预期的小口径差, 跨阶段对比时以此解释。
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

log "==== 四阶段 TopoFR+IR200 v2 训练启动 | campaign=${CAMPAIGN} | sr=0.40 bs=128 | s4 冻结分类器 | ${NGPU} 卡 | 输出根: ${SAVE_ROOT} ===="

################################################################################
# s1: 分类器训练 (backbone 全冻结, step 调度, 5 epoch)
#     PFC 随机初始化, 对 Glint360K TopoFR 预训练 backbone 重新拟合分类头
################################################################################
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
# s2: body.72~97+output_layer 训练 + 分类器 0.1x lr 同步微调 (cosine, 15 epoch)
#     冻结段 BN 维持 eval, 解冻段 BN 随 train 更新 (pipeline train() 内建)
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body72" 15 \
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
# s3: 分类器再训练 (backbone 全冻结, step 调度, 5 epoch), 承接 s2 后的 backbone
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
# s4: 全模型微调, 分类器冻结 (cosine, 15 epoch) 【v2 核心差异】
#     classifiers.freeze=True: 分类器由 pefts.classifier_ckpt_dir 从 s3 加载后
#     全程 requires_grad=False, 不进优化器 (optims.py 对无可训练参数的分类器
#     不加参数组, 故不设 classifier_lr_scale); 仅作固定度量锚参与前向 AdaFace
#     margin loss (batch_mean/std 自适应仍随前向更新)。backbone 向固定分类器
#     对齐, 避免 0.1x lr 同步微调中两方更新互相拉扯的度量漂移。
#     显存: 无分类器权重梯度/动量, 每卡略有富余; bs 仍锁 128 (拓扑 O(N²) 约束)。
################################################################################
run_stage s4 "${CAMPAIGN}_s4_full" 15 \
    "${COMMON[@]}" \
    models.start_from="${S3_DIR}/model.pt" \
    models.freeze=False \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/full.yaml \
    pefts.classifier_ckpt_dir="${S3_DIR}" \
    classifiers.freeze=True \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0008 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler=cosine
S4_DIR="${LAST_STAGE_DIR}"

log "==== 四阶段 TopoFR+IR200 v2 (s4 冻结分类器) 训练全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final --timeout_minutes 90"

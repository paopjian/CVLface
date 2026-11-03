#!/usr/bin/env bash
################################################################################
# 四阶段微调训练 — IR-200 + QGFace + SubCenter 变体 (work_260922_ir200_qgface_subcenter)
#
# 组合来源:
#   - work_260922_ir200_subcenter: IR-200 backbone (insightface 风格 [6,26,60,6]=98 块,
#     118.8M 参数) + 自产 IR200-glint 预训练 + body.72 解冻点 + 显存基线
#   - work_260922_qgface_subcenter: QGFace 三件套 (双视图增强 + AdaFace 质量门控 +
#     TrainQGFacePipeline 双优化器)
#
# 相对 four_stage_qgface_subcenter.sh 的差异 (IR200 化):
#   - backbone: iResNet-200 insightface 风格, models=iresnet_insightface/configs/v1_ir200.yaml
#     (pefts/__init__.py 已带入 ir100/ir200 的 body.K -> layerX.Y 映射)
#   - 预训练起点: 自产 IR200-glint (try_i200 2026-10-18, 26 epoch 从头训: agedb 98.22 /
#     tinyface r1 72.83 / IJBC@1e-5 94.77), md5=10206f92610c1305db5a2bd12914dd78
#   - s2 解冻点: body.72 (= layer1.6+layer2.26+layer3 前 40 块之后, ~73% 深度处,
#     对应 ir101 的 body.36)
#   - bs 按双视图折算对齐 ir200_subcenter 的实测显存基线 (ir200 解冻段激活 ~1.7x ir101):
#       s1/s3: bs 256 (双视图=512 图/卡, 等同 ir200_subcenter s1/s3 bs512 单视图)
#       s2:    bs 128 (双视图=256 图/卡, 等同 ir200_subcenter s2 bs256 单视图)
#       s4:    bs 64  (双视图=128 图/卡, 等同 ir200_subcenter s4 bs128 单视图)
#
# 【本变体核心设计】s2 分类器冻结, 不跟随训练:
#   - classifiers.freeze=True: PFC 权重 (K=3 子中心) 固定为 s1 训完的状态,
#     make_qgface_optimizers 不再创建 classifier_optimizer (无参数更新)
#   - 分类损失梯度仍经冻结的分类器回传到 backbone (embeddings 有梯度, 权重无)
#   - QG 对比损失的队列 proxy (get_class_proxies) 读冻结权重, 整个 s2 固定;
#     子中心路由 (route_subcenters) 仍由原图视图特征实时决定
#   - s3 再训练分类器消化 s2 挪动过的 backbone 特征空间
#
# 【显存注意】
#   - 双视图使每卡过 backbone 图像数 = 2 x batch_size, 上表已按此折算。
#     若仍 OOM: 先减半 trainers.batch_size, 再降 classifiers.sample_rate。
#   - K=3 + sample_rate=0.40 使每卡激活中心行数 x3 (~12.4万行), logit 显存/耗时
#     高于 K=1 基线。
#   - QGFace 约束: pipelines.qgface.queue_size(8192) >= 2 x 全局 batch
#     (s1/s3: 256x8x2=4096; s2: 128x8x2=2048; s4: 64x8x2=1024; 均 <= 8192 ✓)。
#   - 分类器断点/衔接全程要求 K=3, 勿与 K=1 campaign 的 checkpoint 目录混用。
#
# 【服务器环境 — 本脚本仅适配此机器, 换机器必改下面三处路径】
#   服务器:   qingsi (192.168.2.103, 8x RTX 4090, 数据在 /data1/dataset_260918)
#   cvlface:  /root/anaconda3/envs/cvlface (torch 2.12+cu126, python 3.12)
#   必改三处: ① 本节 export 的环境路径 ② trainers/configs/default.yaml 的
#             external_eval_fabric_bin ③ WORKDIR/PRETRAINED 等绝对路径
#
# 【勿开 expandable_segments】外挂评估的 v7 相似度 JIT 扩展(fused_he_pn_v1)与
#   expandable_segments 的 VA 布局不兼容, 训练内评估必崩 CUDA illegal memory access
#   (2026-10-18 排查: 手动无此变量评估成功/训练内继承此变量必崩)。
#   GPU0 碎片化风险由 bs 预留余量覆盖。
#
# 模型:   iResNet-200 (insightface, IR200-glint 预训练 [自产])
# 数据:   /data1/dataset_260918/train_rec (RecordIO, 823875 类, 38.8M 图)
# 输出:   /data1/dataset_260918/train_output/<prefix>_MM-DD_N/checkpoints_every_epoch/epoch:N
#
# 阶段设计:
#   s1  分类器训练(K=3随机初始化), backbone 冻结, QG关闭  step  lr=0.008, 5  epoch, bs 256
#   s2  body.72~97+output_layer, 分类器冻结(本变体), QG开启  cos  lr=0.008, 11  epoch, bs 128
#   s3  分类器再训练, backbone 冻结, QG关闭                 step  lr=0.006, 5  epoch, bs 256
#   s4  全模型训练 + 分类器 0.1x lr, QG开启                 cos  lr=0.0004, 15 epoch, bs 64
#
# 【s2 提前裁定记录 2026-10-25】原设计 15 epoch。epoch10 评估后早停监视器触发
#   (基准被 epoch0 的异常高点 46.17 锁死, 实际指标 epoch5 后触底回升), 早停广播
#   bug 导致 rc=1 + 自动重试, 重试 run (10-25_0) 跑出 epoch:11 且指标全面更好
#   (34t@1e-6: 68.7→79.9)。人工裁定 s2 止于 epoch:11: 本脚本 s2 目标改为 12
#   (epoch >= 11 即完成), s3 从 10-25_0/epoch:11 衔接。cosine lr 衰减未走完属
#   已知取舍。
#
# 断点续传 (直接重跑本脚本即可, 幂等):
#   - 阶段内: 扫描该 prefix 的所有 run 目录, 取"全局最大完整 epoch"(pipeline.pt+
#     model.pt+classifier_rank7.pt 齐全, pipeline.pt 由 rank0 最后写入即完整性标记),
#     通过 trainers.resume 恢复 model/classifier/qgface_loss/优化器/调度器, 从 epoch+1
#     继续。注意每次重启 fabric 会新建 run 目录, 旧目录保留, 属预期。
#     (s2 分类器冻结时无 classifier_optimizer, pipeline.pt 的 module_names_list 不变,
#      仍含 classifier 本体权重, 断点文件齐全性判定不受影响)
#   - 阶段间: s2/s3/s4 自动读取上一阶段最后一个 epoch
#     (models.start_from=上阶段/model.pt 加载 backbone, pefts.classifier_ckpt_dir
#     加载分类器; 优化器/调度器/QG队列按新阶段重置)。
#   - 阶段完成判定: 存在完整 epoch:$(N-1)。早停等导致未达标时会中止并提示。
#
# 用法:  bash scripts/train/four_stage_ir200_qgface_subcenter.sh   (建议放 tmux/nohup 里)
# 日志:  ${SAVE_ROOT}/shell_logs/ 下每阶段一个文件 (tee 同时输出到终端)
################################################################################
set -uo pipefail

# ----------------------------- 可调参数 -----------------------------
CAMPAIGN="ir200_qgface_subcenter_260922"   # 战役前缀; 换超参数开新战役时改它, 避免误续旧目录
NGPU=8                         # GPU 数 (分类器 partial_fc 按 rank 分片, 中途不可改)
MAX_RETRY=3                    # 阶段内连续无进展的最大重试次数
WORKDIR=/root/zhaokj/CVLface/cvlface/research/recognition/code/work_260922_ir200_qgface_subcenter
SAVE_ROOT=/data1/dataset_260918/train_output
PRETRAINED=/data2/zkj-data/ir200_glint_pretrain/model.pt

# ----------------------------- 环境 (qingsi 192.168.2.103, 8 卡) -----------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/root/anaconda3/envs/cvlface
export PATH=/root/anaconda3/envs/cvlface/bin:$PATH
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}   # scipy libstdc++ 冲突
export DECODE_BACKEND=turbojpeg
# 【wandb 禁用 2026-11-01】API key 失效 (服务端 403, AuthenticationError), wandb.init
# 必炸: rank0 抛 CommError 退出, ranks1-7 NCCL 空转 2h。已改用 MLflow 记录
# (train_opt.py 内建, per-run SQLite: experiments/<run>/mlflow.db, mlflow ui 查看)。
export WANDB_MODE=disabled

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
# IR200 backbone (insightface 风格, 与 TreB1eN 风格权重不互通) + QGFace 三件套
# (双视图增强 + adaface 质量门控 + TrainQGFacePipeline, 对比权重逐阶段用
#  pipelines.contrast_weight 覆盖) + SubCenter PFC K=3。
# epoch 末评估走外挂 TRT 单进程链路 (eval_all_trt_single: nvjpeg 解码 + TRT fp16
# + v7 匹配, 无 NCCL/fabric); TRT fp16 与进程内 torch bf16 评估存在可预期的小口径差。
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
    losses=configs/qgface_adaface.yaml
    data_augs=configs/qgface.yaml
    pipelines=configs/train_qgface.yaml
    evaluations=configs/val_20260922.yaml
    trainers.skip_final_eval=True
)

log "==== 四阶段 IR200+QGFace+SubCenter 训练启动 | campaign=${CAMPAIGN} | K=3 sr=0.40 | s2 分类器冻结 | ${NGPU} 卡 | 预训练: ${PRETRAINED} ===="

################################################################################
# s1: 分类器训练 (SubCenter K=3 随机初始化, backbone 全冻结, QG 关闭, step 调度, 5 epoch)
#     双视图仍进分类损失 (contrast_weight=0 只关对比项)
#     bs 256 (双视图=512 图/卡, 等同 ir200_subcenter s1 bs512 单视图的每卡负载)
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
# s2: body.72~97+output_layer 训练, 分类器冻结 (本变体核心设计), QG 开启 (cosine)
#     - body.72 = ir200 的 ~73% 深度 (对应 ir101 的 body.36); pefts 映射已带入
#     - classifiers.freeze=True: PFC 权重固定为 s1 状态, 不创建 classifier_optimizer;
#       分类损失梯度仍回传 backbone; QG 队列 proxy 读冻结权重 (整阶段固定),
#       子中心路由仍由原图视图特征实时决定
#     - bs 128 (双视图=256 图/卡, 对齐 ir200_subcenter s2 bs256 单视图的实测显存;
#       ir200 解冻段激活 ~1.7x ir101, 不可沿用 ir101 qgface 的 bs256)
#     - 目标 12 = 人工裁定 epoch:11 为 s2 终点 (见文件头裁定记录), 非原设计 15
################################################################################
run_stage s2 "${CAMPAIGN}_s2_body72_clsfrz" 12 \
    trainers.batch_size=128 \
    "${COMMON[@]}" \
    models.start_from="${S1_DIR}/model.pt" \
    models.freeze=True \
    pipelines.contrast_weight=1.0 \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.72 \
    pefts.classifier_ckpt_dir="${S1_DIR}" \
    classifiers.freeze=True \
    optims=configs/qgface_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=15 optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.max_grad_norm=5.0 \
    optims.scheduler=cosine
S2_DIR="${LAST_STAGE_DIR}"

################################################################################
# s3: 分类器再训练 (backbone 全冻结, QG 关闭, step 调度, 5 epoch), 承接 s2 后的 backbone
#     s2 期间分类器被冻结, s3 重新训练 K=3 子中心以对齐 s2 挪动过的特征空间
#     bs 256 (双视图=512 图/卡, 等同 ir200_subcenter s3 bs512 单视图的每卡负载)
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
#     bs 64 (双视图=128 图/卡, 对齐 ir200_subcenter s4 bs128 单视图的每卡负载;
#     兼顾 subcenter 工作区 OOM 教训 提交 d66022d: K=3+sr0.4 每卡 256 图激活超 24G)
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

log "==== 四阶段 IR200+QGFace+SubCenter 训练全部完成 ===="
log "最终 checkpoint: ${S4_DIR}"
log "最终测试评估示例:"
log "  python eval_all_trt_launcher.py --num_gpu ${NGPU} --eval_config_name test_20260922 \\"
log "      --ckpt_dir ${S4_DIR} --project_name ${CAMPAIGN} --name final --timeout_minutes 90"

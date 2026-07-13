#!/usr/bin/env bash
# QCFace IR-101 两阶段训练 (dataset_0605, 791k classes)
#
# 阶段1 (5 epoch): backbone 全冻结, 只训分类器 (让分类器充分收敛)
#   - losses.warmup_id_only_epochs=5 与阶段1的5个epoch自然对齐:
#     整个阶段1期间 norm loss 关闭, 只用 ID loss 热身分类器
# 阶段2 (13 epoch): 渐进式解冻 (只解冻 body.36 之后), backbone + 分类器训练
#   - 从阶段1 checkpoint 加载 backbone 和分类器权重
#   - pefts=part_freeze target=body.36: 只解冻后半段 block, 避免全解冻冲垮
#     97.7% agedb 的强预训练 backbone
#   - losses.warmup_id_only_epochs=0: 分类器已收敛, 立即开启 norm loss
#
# 使用方法 (全自动, 直接 bash 即可, 会连续跑完两个阶段):
#   bash run_qcface_ir101_0605_2stage.sh
#     - 未检测到阶段1 checkpoint  → 先跑阶段1, 完成后自动进入阶段2
#     - 已检测到阶段1 checkpoint  → 直接跑阶段2
#
# 也可显式指定阶段 (强制只跑该阶段, 不自动串联):
#   bash run_qcface_ir101_0605_2stage.sh stage1
#   bash run_qcface_ir101_0605_2stage.sh stage2

set -e

SAVE_DIR="/data1/dataset_0605/train_output"
STAGE1_PREFIX="qcface_ir101_0605_s1_cls"
STAGE1_EPOCHS=5   # 阶段1 num_epoch, checkpoint 为 epoch:$((STAGE1_EPOCHS-1))

export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export FINAL_EVAL_CONFIG=test_20260605

# ─────────────────────────────────────────────────────────────────────────────
# 自动探测阶段1的最终 checkpoint (epoch:$((STAGE1_EPOCHS-1)), 去掉 step 后命名固定)
# glob 所有匹配的 run 目录, 按修改时间取最新, 并校验 model.pt/classifier.pt 存在
# ─────────────────────────────────────────────────────────────────────────────
detect_stage1_ckpt() {
    local last_epoch=$((STAGE1_EPOCHS - 1))
    local ckpt
    ckpt="$(ls -dt "${SAVE_DIR}/${STAGE1_PREFIX}"_*/checkpoints_every_epoch/epoch:${last_epoch}/ 2>/dev/null | head -1)"
    ckpt="${ckpt%/}"
    if [[ -n "$ckpt" && -f "${ckpt}/model.pt" && -f "${ckpt}/classifier_rank0.pt" ]]; then
        echo "$ckpt"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 公共参数
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    trainers.num_gpu=8
    trainers.batch_size=256
    trainers.gradient_acc=1
    trainers.num_workers=8
    trainers.precision=bf16-mixed
    trainers.float32_matmul_precision=high
    trainers.using_wandb=True
    models=iresnet_qcface/configs/v1_ir101_qcface.yaml
    dataset=configs/dataset_0605_train_rec.yaml
    dataset.model_save_dir=/data1/dataset_0605/train_output
    data_augs=configs/basic_v2_numpy.yaml
    pipelines=configs/train_model_cls.yaml
    classifiers=configs/partial_fc.yaml
    losses=configs/qcface.yaml
    optims=configs/step_sgd.yaml
    evaluations=configs/val_20260605.yaml
)

# ─────────────────────────────────────────────────────────────────────────────
# 阶段1: 分类器预热 (backbone 全冻结, 5 epoch)
#
# backbone 加载 qcface.pth 预训练权重 (在 v1_ir101_qcface.yaml 中已指定)
# pefts=freeze: 冻结所有 backbone 参数 (requires_grad=False, 不参与反向传播)
# warmup_id_only_epochs=5: 与 num_epoch=5 对齐, 整个阶段只用 ID loss
#
# lr/schedule 调整: 让分类器充分收敛 (原 [2,4] 过早 decay 导致 loss 卡在 ~17)
#   lr=0.02, milestones=[3,4] 靠后, 保证前 3 epoch 全速学习
# ─────────────────────────────────────────────────────────────────────────────
run_stage1() {
echo "========================================="
echo "阶段1: 分类器预热 (backbone 冻结, ${STAGE1_EPOCHS} epoch)"
echo "========================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
    --devices=8 \
    --precision="bf16-mixed" \
    train_opt.py \
    "${COMMON_ARGS[@]}" \
    trainers.prefix=qcface_ir101_0605_s1_cls \
    trainers.batch_size=512 \
    pefts=configs/freeze.yaml \
    optims.lr=0.02 \
    optims.num_epoch=${STAGE1_EPOCHS} \
    "optims.lr_milestones=[3,4]" \
    optims.lr_lambda=0.3 \
    optims.warmup_epoch=1 \
    optims.momentum=0.9 \
    optims.weight_decay=0.0001 \
    losses.warmup_id_only_epochs=${STAGE1_EPOCHS} \
    trainers.skip_final_eval=True
}

# ─────────────────────────────────────────────────────────────────────────────
# 阶段2: 渐进式解冻训练 (只解冻 body.36 之后, 13 epoch)
#
# models.start_from: 加载阶段1的 backbone 权重 (model.pt 单文件)
# classifiers.start_from: 加载阶段1的分类器权重 (传目录, 内含 classifier_rank*.pt 分片)
# pefts=part_freeze target_modules=qcface_layer4: 只解冻 net.layer4 + 输出头
#   —— QCFace iresnet 结构为 net.layer1..layer4, 没有 baseline 的 body.N 命名;
#      qcface_layer4 映射在 pefts/__init__.py 中定义 (可选 qcface_tail/layer3/layer2)
#   —— 只解冻后半段, 避免全解冻在 1 epoch 内冲垮 97.7% agedb 的强预训练
# warmup_id_only_epochs=0: 分类器已收敛, 立即启用 norm loss (loss_g)
# lr=0.005: 比原 0.01 更低, 保护强初始权重; milestones=[3,7,10]/13epoch
# ─────────────────────────────────────────────────────────────────────────────
run_stage2() {
    local stage1_ckpt="$1"
    if [[ -z "$stage1_ckpt" ]]; then
        echo ""
        echo "ERROR: 未找到阶段1 checkpoint, 无法运行阶段2"
        echo "  搜索路径: ${SAVE_DIR}/${STAGE1_PREFIX}_*/checkpoints_every_epoch/epoch:$((STAGE1_EPOCHS-1))/"
        echo "  请先运行: bash $(basename "$0") stage1"
        exit 1
    fi

    echo ""
    echo "========================================="
    echo "阶段2: 渐进式解冻 (body.36+, 13 epoch)"
    echo "========================================="
    echo "加载 backbone: ${stage1_ckpt}/model.pt"
    echo "加载分类器:    ${stage1_ckpt}/ (classifier_rank*.pt 分片)"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
        --devices=8 \
        --precision="bf16-mixed" \
        train_opt.py \
        "${COMMON_ARGS[@]}" \
        trainers.prefix=qcface_ir101_0605_s2_full \
        models.start_from="${stage1_ckpt}/model.pt" \
        classifiers.start_from="${stage1_ckpt}" \
        pefts=configs/part_freeze.yaml pefts.target_modules=qcface_layer4 \
        optims.lr=0.005 \
        optims.num_epoch=13 \
        "optims.lr_milestones=[3,7,10]" \
        optims.lr_lambda=0.1 \
        optims.warmup_epoch=1 \
        optims.momentum=0.9 \
        optims.weight_decay=0.0001 \
        losses.warmup_id_only_epochs=0 \
        trainers.skip_final_eval=False
}

# ─────────────────────────────────────────────────────────────────────────────
# 调度逻辑
#   无参数: 全自动 —— 无 ckpt 则跑完阶段1再自动进入阶段2; 有 ckpt 则直接阶段2
#   stage1 / stage2: 只跑指定阶段
# ─────────────────────────────────────────────────────────────────────────────
STAGE="$1"

case "$STAGE" in
    stage1)
        run_stage1
        echo ""
        echo "阶段1 完成 (仅阶段1模式)。运行 stage2: bash $(basename "$0") stage2"
        ;;
    stage2)
        run_stage2 "$(detect_stage1_ckpt)"
        ;;
    stage2_from_pretrain)
        # 跳过 stage1，直接从原始预训练权重启动 stage2。
        # 适用场景：旧 stage1 ckpt 的 BN running stats 已被污染，不可复用。
        # backbone 由 v1_ir101_qcface.yaml 里的 start_from 直接加载预训练权重，
        # 分类器随机初始化（classifiers.start_from 为空）。
        echo "========================================="
        echo "stage2_from_pretrain: 直接从预训练权重开始"
        echo "  backbone : /root/zhaokj/QCFace/pretrained/qcface.pth (via model yaml)"
        echo "  分类器   : 随机初始化"
        echo "========================================="
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
            --devices=8 \
            --precision="bf16-mixed" \
            train_opt.py \
            "${COMMON_ARGS[@]}" \
            trainers.prefix=qcface_ir101_0605_s2_pretrain \
            pefts=configs/part_freeze.yaml pefts.target_modules=qcface_layer4 \
            optims.lr=0.005 \
            optims.num_epoch=13 \
            "optims.lr_milestones=[3,7,10]" \
            optims.lr_lambda=0.1 \
            optims.warmup_epoch=1 \
            optims.momentum=0.9 \
            optims.weight_decay=0.0001 \
            losses.warmup_id_only_epochs=5 \
            trainers.skip_final_eval=False
        ;;
    "")
        # 全自动
        STAGE1_CKPT="$(detect_stage1_ckpt)"
        if [[ -z "$STAGE1_CKPT" ]]; then
            echo "[自动调度] 未检测到阶段1 checkpoint → 先跑阶段1"
            run_stage1
            STAGE1_CKPT="$(detect_stage1_ckpt)"
            echo "[自动调度] 阶段1 完成 → 自动进入阶段2"
        else
            echo "[自动调度] 检测到阶段1 checkpoint → 直接运行阶段2"
            echo "           $STAGE1_CKPT"
        fi
        run_stage2 "$STAGE1_CKPT"
        ;;
    *)
        echo "用法: bash $(basename "$0") [stage1|stage2|stage2_from_pretrain]"
        echo "  不带参数            : 全自动 (无 ckpt→跑阶段1后自动进阶段2, 有 ckpt→直接阶段2)"
        echo "  stage2_from_pretrain: 跳过 stage1, 直接从预训练权重开始 stage2 (旧 ckpt BN 污染时使用)"
        exit 1
        ;;
esac

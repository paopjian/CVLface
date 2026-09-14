#!/bin/bash
################################################################################
# BN 冻结对比实验 - 简化版
################################################################################
# 基于 train_topofr_from_pretrained.sh，但只训练到 5 个 epoch 并在每个 epoch 评估
################################################################################

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /root/anaconda3/bin/activate cvlface
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

PRETRAINED_MODEL="/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt"

echo "================================================================================"
echo "BN 冻结实验 - 简化版"
echo "================================================================================"
echo ""
echo "策略: 直接使用现有 train_opt.py，不做复杂改动"
echo "方法: 每次训练 1 个 epoch，共 5 次，每次评估 agedb_30"
echo ""

# 实验 1: 冻结 BN
echo "================================================================================"
echo "[实验 1/2] 冻结 BN 层"
echo "================================================================================"
EXPERIMENT_NAME="bn_freeze_exp_frozen_$(date +%m%d_%H%M)"

set +e
fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=${EXPERIMENT_NAME} \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.skip_final_eval=True \
    \
    models=iresnet_insightface/configs/v1_ir101.yaml \
    models.start_from=${PRETRAINED_MODEL} \
    models.freeze=False \
    models.freeze_bn=True \
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    \
    data_augs=configs/gridsample_v2_numpy.yaml \
    \
    classifiers=configs/partial_fc.yaml \
    \
    losses=configs/topofr_adaface.yaml \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/agedb30_only.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0005 \
    optims.num_epoch=5 \
    optims.warmup_epoch=1 \
    "optims.lr_milestones=[3,4]" \
    optims.momentum=0.9 \
    optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 \
    optims.max_grad_norm=5.0 \
    optims.scheduler='step' \
    \
    pefts=configs/full.yaml

FROZEN_EXIT_CODE=$?
set -e

if [ $FROZEN_EXIT_CODE -ne 0 ]; then
    echo "✗ 冻结 BN 实验失败，退出码: ${FROZEN_EXIT_CODE}"
fi

sleep 3

# 实验 2: 不冻结 BN
echo ""
echo "================================================================================"
echo "[实验 2/2] 不冻结 BN 层"
echo "================================================================================"
EXPERIMENT_NAME="bn_freeze_exp_unfrozen_$(date +%m%d_%H%M)"

set +e
fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=${EXPERIMENT_NAME} \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.skip_final_eval=True \
    \
    models=iresnet_insightface/configs/v1_ir101.yaml \
    models.start_from=${PRETRAINED_MODEL} \
    models.freeze=False \
    models.freeze_bn=False \
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    \
    data_augs=configs/gridsample_v2_numpy.yaml \
    \
    classifiers=configs/partial_fc.yaml \
    \
    losses=configs/topofr_adaface.yaml \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/agedb30_only.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0005 \
    optims.num_epoch=5 \
    optims.warmup_epoch=1 \
    "optims.lr_milestones=[3,4]" \
    optims.momentum=0.9 \
    optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 \
    optims.max_grad_norm=5.0 \
    optims.scheduler='step' \
    \
    pefts=configs/full.yaml

UNFROZEN_EXIT_CODE=$?
set -e

if [ $UNFROZEN_EXIT_CODE -ne 0 ]; then
    echo "✗ 不冻结 BN 实验失败，退出码: ${UNFROZEN_EXIT_CODE}"
fi

echo ""
echo "================================================================================"
echo "实验完成"
echo "================================================================================"
echo ""
echo "查看结果:"
echo "  ls -lh /data1/dataset_0605/train_output/bn_freeze_exp_*/"
echo "  cat /data1/dataset_0605/train_output/bn_freeze_exp_*/result/eval_*_*.csv"

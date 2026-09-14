#!/bin/bash
################################################################################
# TopoFR 快速验证 - 使用 TopoFR 预训练模型
# 用于快速验证代码和配置
################################################################################

set -e

echo "================================================================================"
echo "TopoFR 快速验证 (TopoFR预训练模型)"
echo "================================================================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
# 791509 类的 FC 头显存吃紧(实测峰值 22.0/23.5 GiB), 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 设为 N 时每 N 步打印 loss_cls / loss_topo 配比, 默认关闭
# export TOPOFR_LOG_EVERY=50

source /root/anaconda3/bin/activate cvlface
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

PRETRAINED_MODEL="/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt"
EXPERIMENT_NAME="topofr_r100_quick_test_$(date +%m%d_%H%M)"

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 预训练模型不存在"
    exit 1
fi

echo "预训练模型: ${PRETRAINED_MODEL}"
echo "模式: 快速验证 (5 epochs)"
echo "实验名称: ${EXPERIMENT_NAME}"
echo ""

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
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    \
    data_augs=configs/basic_v2_numpy.yaml \
    \
    classifiers=configs/fc.yaml \
    \
    losses=configs/topofr_adaface.yaml \
    \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/val_20260605.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0005 \
    optims.num_epoch=5 \
    optims.warmup_epoch=1 \
    "optims.lr_milestones=[3]" \
    optims.momentum=0.9 \
    optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 \
    optims.max_grad_norm=5.0 \
    optims.scheduler='step' \
    \
    pefts=configs/full.yaml

echo ""
echo "================================================================================"
echo "快速验证完成！"
echo "输出: /data1/dataset_0605/train_output/${EXPERIMENT_NAME}"
echo "================================================================================"

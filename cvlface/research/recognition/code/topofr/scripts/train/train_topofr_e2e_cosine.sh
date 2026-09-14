#!/bin/bash
################################################################################
# TopoFR 端到端全量训练 - 使用 Cosine LR 调度器
# 更平滑的学习率衰减，适合长时间训练
################################################################################

set -e

echo "================================================================================"
echo "TopoFR 端到端全量训练 (Cosine LR)"
echo "================================================================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
# 791509 类 FC 头显存吃紧(实测峰值 22.0/23.5 GiB), 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /root/anaconda3/bin/activate cvlface
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

EXPERIMENT_NAME="topofr_e2e_cosine_$(date +%m%d_%H%M)"

echo "实验名称: ${EXPERIMENT_NAME}"
echo "学习率调度: Cosine Annealing"
echo "总epochs: 30"
echo ""

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=${EXPERIMENT_NAME} \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.skip_final_eval=False \
    \
    models=iresnet/configs/v1_ir101.yaml \
    models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
    models.freeze=False \
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    \
    data_augs=configs/gridsample_v2_numpy.yaml \
    \
    classifiers=configs/fc.yaml \
    classifiers.sample_rate=0.40 \
    \
    losses=configs/topofr_adaface.yaml \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/val_20260605.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.001 \
    optims.num_epoch=30 \
    optims.warmup_epoch=3 \
    optims.momentum=0.9 \
    optims.weight_decay=0.0005 \
    optims.max_grad_norm=5.0 \
    optims.scheduler='cosine' \
    \
    pefts=configs/full.yaml

echo ""
echo "================================================================================"
echo "训练完成！输出目录: /data1/dataset_0605/train_output/${EXPERIMENT_NAME}"
echo "================================================================================"

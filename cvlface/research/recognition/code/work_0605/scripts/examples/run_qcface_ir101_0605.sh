#!/usr/bin/env bash
# Run QCFace (base) with QCFace-original iResNet-101 on dataset_0605 (791k classes).
#
# Backbone:  iresnet_qcface/configs/v1_ir101_qcface.yaml
#            (QCFace's own IResNet naming, loaded from qcface.pth pretrained weights)
# Loss:      losses=configs/qcface.yaml  — ArcFace hard margin + magnitude-reg
# Schedule:  12 epoch, lr drop at 5/8/10 (mirrors QCFace paper's MS1MV3 setup)
# WandB:     logs train/qcface_loss_id, train/qcface_loss_g, train/qcface_mean_norm

export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
# 最终评估使用 test_20260605 测试集
export FINAL_EVAL_CONFIG=test_20260605

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
    --devices=8 \
    --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=qcface_ir101_0605 \
    trainers.num_gpu=8 \
    trainers.batch_size=256 \
    trainers.gradient_acc=1 \
    trainers.num_workers=8 \
    trainers.precision='bf16-mixed' \
    trainers.float32_matmul_precision='high' \
    trainers.using_wandb=True \
    models=iresnet_qcface/configs/v1_ir101_qcface.yaml \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    data_augs=configs/basic_v2_numpy.yaml \
    pipelines=configs/train_model_cls.yaml \
    classifiers=configs/partial_fc.yaml \
    losses=configs/qcface.yaml \
    optims=configs/step_sgd.yaml \
    optims.num_epoch=18 \
    optims.lr=0.01 \
    optims.lr_milestones="[8,12,15]" \
    optims.warmup_epoch=1 \
    evaluations=configs/val_20260605.yaml \
    pefts=configs/none.yaml \
    trainers.skip_final_eval=True

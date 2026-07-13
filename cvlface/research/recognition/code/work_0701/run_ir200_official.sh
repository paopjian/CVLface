#!/bin/bash
# IR-200 从头训练 (仿 AdaFace 官方配方: SGD + step衰减 + lr0.1)
# 数据: train_rec2 (682039类/35M图), bs=64/卡×8 = total batch 512 (与官方一致)
# 调度: warmup 2 epoch, 共 18 epoch, milestones [8,14,17] (由官方[12,20,24]/26 等比缩放)
# 评估: 每 epoch 用 val_20260605 (9个验证集), final eval 关闭
set -e
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/work_0701

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run -n cvlface --no-capture-output bash -c "
  fabric run --devices=8 --precision=bf16-mixed train_opt.py \
    trainers.prefix=ir200_scratch_official trainers.num_gpu=8 trainers.batch_size=64 trainers.num_workers=8 \
    trainers.precision=bf16-mixed trainers.using_wandb=True \
    trainers.skip_final_eval=True \
    models=iresnet/configs/v1_ir200.yaml models.start_from='' \
    dataset=configs/dataset_0605_train_rec2.yaml data_augs=configs/gridsample_v2_numpy.yaml \
    classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.1 \
    losses=configs/adaface.yaml pefts=configs/full.yaml \
    optims=configs/step_sgd.yaml optims.num_epoch=18 optims.warmup_epoch=2 optims.lr=0.1 \
    optims.lr_milestones='[8,14,17]' \
    evaluations=configs/val_20260605.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output
"

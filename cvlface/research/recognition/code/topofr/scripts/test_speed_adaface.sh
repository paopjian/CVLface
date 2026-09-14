#!/bin/bash
################################################################################
# 快速测试：对比 AdaFace vs TopoFR 的训练速度
# 只跑 200 步，看实际速度差异
################################################################################

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /root/anaconda3/bin/activate cvlface
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

PRETRAINED_MODEL="/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt"
EXPERIMENT_NAME="speed_test_adaface_$(date +%m%d_%H%M)"

echo "================================================================================"
echo "速度测试：纯 AdaFace (无拓扑损失)"
echo "================================================================================"

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=${EXPERIMENT_NAME} \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.limit_num_batch=200 \
    trainers.skip_final_eval=True \
    \
    models=iresnet_insightface/configs/v1_ir101.yaml \
    models.start_from=${PRETRAINED_MODEL} \
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
    losses=configs/adaface.yaml \
    pipelines=configs/train_model_cls \
    \
    evaluations=configs/skip_eval.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0005 \
    optims.num_epoch=1 \
    \
    pefts=configs/full.yaml

echo "================================================================================"
echo "测试完成！查看最后几行的 Speed 数值"
echo "================================================================================"

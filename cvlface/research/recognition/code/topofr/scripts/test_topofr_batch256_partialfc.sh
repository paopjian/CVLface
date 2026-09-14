#!/bin/bash
################################################################################
# 测试 TopoFR batch=256 + partial_fc 的速度（与历史配置一致）
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
echo "测试 TopoFR: batch=256 + partial_fc（与历史 AdaFace 配置一致）"
echo "================================================================================"

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=topofr_batch256_partialfc_test \
    trainers.num_gpu=8 \
    trainers.batch_size=256 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.limit_num_batch=100 \
    trainers.skip_final_eval=True \
    \
    models=iresnet_insightface/configs/v1_ir101.yaml \
    models.start_from=${PRETRAINED_MODEL} \
    models.freeze=False \
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/tmp/topofr_batch256_test \
    \
    data_augs=configs/basic_v2_numpy.yaml \
    \
    classifiers=configs/partial_fc_sample10.yaml \
    classifiers.sample_rate=0.40 \
    \
    losses=configs/topofr_adaface.yaml \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/skip_eval.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0005 \
    optims.num_epoch=1 \
    \
    pefts=configs/full.yaml \
    2>&1 | tee /tmp/topofr_batch256_partialfc.log

echo ""
echo "================================================================================"
echo "测试完成！"

speed=$(grep -oP 'Speed \K[0-9]+' /tmp/topofr_batch256_partialfc.log | tail -20 | awk '{sum+=$1; count++} END {printf "%.0f", sum/count}')
time=$(grep "100/100" /tmp/topofr_batch256_partialfc.log | tail -1 | grep -oP '\d+:\d+<' | head -1 | tr -d '<')

echo "TopoFR batch=256 + partial_fc:"
echo "  平均速度: ${speed} samples/s ($(echo "scale=2; $speed / 2048" | bc) it/s)"
echo "  100步耗时: ${time}"

if [[ $time =~ ([0-9]+):([0-9]+) ]]; then
    secs=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
    per_step=$(echo "scale=2; $secs / 100" | bc)
    echo "  每步耗时: ${per_step}秒"

    echo ""
    echo "对比："
    echo "  work_0605 AdaFace (batch=256, partial_fc): 7200 samples/s, 3.50 it/s"
    echo "  topofr TopoFR (batch=256, partial_fc):     ${speed} samples/s, $(echo "scale=2; $speed / 2048" | bc) it/s"
    echo "  差异: $(echo "scale=2; 7200 / $speed" | bc)x"
fi

echo "================================================================================"

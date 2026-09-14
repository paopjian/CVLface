#!/bin/bash
################################################################################
# 对比测试：TopoFR vs 纯 AdaFace 的速度差异
# 完全相同的配置，只是 loss 不同
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
echo "速度对比测试：TopoFR vs 纯 AdaFace"
echo "配置：batch=128, 8 GPU, 100步"
echo "================================================================================"

# 1. 测试 TopoFR
echo ""
echo "1. 测试 TopoFR (AdaFace + 拓扑损失)..."
echo "================================================================================"

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=compare_topofr \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
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
    dataset.model_save_dir=/tmp/compare_output \
    \
    data_augs=configs/gridsample_v2_numpy.yaml \
    \
    classifiers=configs/fc.yaml \
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
    2>&1 | tee /tmp/compare_topofr.log

echo ""
echo "TopoFR 测试完成！提取速度..."
topofr_speed=$(grep -oP 'Speed \K[0-9]+' /tmp/compare_topofr.log | tail -20 | awk '{sum+=$1; count++} END {printf "%.0f", sum/count}')
topofr_time=$(grep "100/100" /tmp/compare_topofr.log | tail -1 | grep -oP '\d+:\d+<' | head -1 | tr -d '<')
echo "  平均速度: ${topofr_speed} samples/s"
echo "  100步耗时: ${topofr_time}"

# 2. 测试纯 AdaFace
echo ""
echo "2. 测试纯 AdaFace (无拓扑损失)..."
echo "================================================================================"

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=compare_adaface \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
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
    dataset.model_save_dir=/tmp/compare_output \
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
    pefts=configs/full.yaml \
    2>&1 | tee /tmp/compare_adaface.log

echo ""
echo "AdaFace 测试完成！提取速度..."
adaface_speed=$(grep -oP 'Speed \K[0-9]+' /tmp/compare_adaface.log | tail -20 | awk '{sum+=$1; count++} END {printf "%.0f", sum/count}')
adaface_time=$(grep "100/100" /tmp/compare_adaface.log | tail -1 | grep -oP '\d+:\d+<' | head -1 | tr -d '<')
echo "  平均速度: ${adaface_speed} samples/s"
echo "  100步耗时: ${adaface_time}"

# 3. 对比结果
echo ""
echo "================================================================================"
echo "对比结果"
echo "================================================================================"
echo ""
echo "TopoFR (AdaFace + 拓扑损失):"
echo "  速度: ${topofr_speed} samples/s"
echo "  耗时: ${topofr_time}"
echo ""
echo "纯 AdaFace (无拓扑损失):"
echo "  速度: ${adaface_speed} samples/s"
echo "  耗时: ${adaface_time}"
echo ""

if [ -n "$topofr_speed" ] && [ -n "$adaface_speed" ]; then
    slowdown=$(echo "scale=2; $adaface_speed / $topofr_speed" | bc)
    echo "速度差异: TopoFR 比 AdaFace 慢 ${slowdown}x"

    # 计算拓扑损失的实际开销
    if [[ $topofr_time =~ ([0-9]+):([0-9]+) ]]; then
        topofr_secs=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
    fi
    if [[ $adaface_time =~ ([0-9]+):([0-9]+) ]]; then
        adaface_secs=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
    fi

    if [ -n "$topofr_secs" ] && [ -n "$adaface_secs" ]; then
        diff_secs=$((topofr_secs - adaface_secs))
        per_step=$(echo "scale=3; $diff_secs / 100" | bc)
        echo "拓扑损失实际开销: ${diff_secs}秒 / 100步 = ${per_step}秒/步"
    fi
fi

echo ""
echo "================================================================================"

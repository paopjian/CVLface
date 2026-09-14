#!/bin/bash
################################################################################
# 真实训练速度测试：测试不同 batch size 的实际速度
# 每个配置跑 100 步
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
echo "真实训练速度测试 - TopoFR"
echo "每个配置跑 100 步，记录实际速度"
echo "================================================================================"

for BATCH_SIZE in 128 64 32 16; do
    TOTAL_BATCH=$((BATCH_SIZE * 8))
    EXPERIMENT_NAME="speed_test_batch${BATCH_SIZE}_$(date +%H%M%S)"

    echo ""
    echo "================================================================================"
    echo "测试: per-GPU batch=${BATCH_SIZE}, 总 batch=${TOTAL_BATCH}"
    echo "================================================================================"

    fabric run --devices=8 --precision="bf16-mixed" \
        train_opt.py \
        trainers.prefix=${EXPERIMENT_NAME} \
        trainers.num_gpu=8 \
        trainers.batch_size=${BATCH_SIZE} \
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
        dataset.model_save_dir=/tmp/speed_test_output \
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
        optims.warmup_epoch=0 \
        \
        pefts=configs/full.yaml \
        2>&1 | tee /tmp/speed_test_batch${BATCH_SIZE}.log

    echo ""
    echo "batch=${BATCH_SIZE} 测试完成，查看平均速度:"
    grep -E "Speed [0-9]+" /tmp/speed_test_batch${BATCH_SIZE}.log | tail -20 | awk '{print $8}' | awk '{sum+=$1; count++} END {print "  平均速度: " sum/count " samples/s (" sum/count/'"${TOTAL_BATCH}"' " it/s)"}'

    # 等待一下，让 GPU 冷却
    sleep 5
done

echo ""
echo "================================================================================"
echo "所有测试完成！汇总结果:"
echo "================================================================================"
for BATCH_SIZE in 128 64 32 16; do
    TOTAL_BATCH=$((BATCH_SIZE * 8))
    echo -n "batch=${BATCH_SIZE} (总=${TOTAL_BATCH}): "
    if [ -f /tmp/speed_test_batch${BATCH_SIZE}.log ]; then
        grep -E "Speed [0-9]+" /tmp/speed_test_batch${BATCH_SIZE}.log | tail -20 | awk '{print $8}' | awk '{sum+=$1; count++} END {printf "%.0f samples/s (%.2f it/s)\n", sum/count, sum/count/'"${TOTAL_BATCH}"'}'
    else
        echo "未完成"
    fi
done
echo "================================================================================"

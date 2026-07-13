#!/bin/bash

LOGDIR="./monitor_logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MEM_LOG="$LOGDIR/mem_${TIMESTAMP}.log"
TRAIN_LOG="$LOGDIR/train_${TIMESTAMP}.log"

echo "Memory log: $MEM_LOG"
echo "Train log:  $TRAIN_LOG"

# 启动内存监控 (每10秒记录一次)
(
    while true; do
        echo "=== $(date '+%H:%M:%S') ===" >> "$MEM_LOG"
        free -h | head -3 >> "$MEM_LOG"
        echo "Swap used: $(swapon --show=USED --noheadings 2>/dev/null || grep SwapTotal /proc/meminfo)" >> "$MEM_LOG"
        # rank0 进程的 VmRSS 和 VmSwap
        PID=$(pgrep -f "train_opt.py" | head -1)
        if [ -n "$PID" ]; then
            echo "Rank0 PID=$PID  RSS=$(awk '/VmRSS/{print $2,$3}' /proc/$PID/status 2>/dev/null)  Swap=$(awk '/VmSwap/{print $2,$3}' /proc/$PID/status 2>/dev/null)" >> "$MEM_LOG"
        fi
        echo "" >> "$MEM_LOG"
        sleep 10
    done
) &
MONITOR_PID=$!

# 启动训练
COMMON="trainers.num_gpu=8 trainers.batch_size=256 trainers.num_workers=8 \
trainers.precision=bf16-mixed \
models=iresnet/configs/v1_ir101.yaml \
models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
dataset=configs/dataset_0605_train_rec2.yaml \
data_augs=configs/basic_v2_numpy.yaml \
classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.40 \
losses=configs/adaface.yaml \
evaluations=configs/val_20260605.yaml \
dataset.model_save_dir=/data1/dataset_0605/train_output"

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    $COMMON \
    trainers.prefix=s2_body36_0605_v2 \
    trainers.num_gpu=8 trainers.batch_size=512 trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    models=iresnet/configs/v1_ir101.yaml \
    models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
    models.freeze=True \
    classifiers.freeze=True \
    data_augs=configs/gridsample_v2_numpy.yaml \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.36 \
    pefts.classifier_ckpt_dir=/data1/dataset_0605/train_output/s1_warmup_0605_v2_06-13_0/checkpoints_every_epoch/epoch:4_step:43070 \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=15 \
    optims.warmup_epoch=2 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 optims.max_grad_norm=5.0 \
    optims.scheduler='cosine' \
    trainers.skip_final_eval=True \
    dataset=configs/dataset_0605_train_rec2.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    2>&1 | tee "$TRAIN_LOG"

# 训练结束后停止监控
kill $MONITOR_PID 2>/dev/null
echo "Done. Logs saved to $LOGDIR/"

#!/bin/bash
# 真实 7卡 DDP 训练 Benchmark
# 对比 PIL vs TurboJPEG 解码后端, 1000 batch, batch_size=512
#
# 用法:
#   conda activate cvlface
#   bash scripts/benchmark/bench_real_train.sh
#
# 监控 CPU/内存: 脚本自动后台记录 vmstat + nvidia-smi

set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH

COMMON="trainers.num_gpu=7 trainers.num_workers=8 \
trainers.precision=bf16-mixed \
models=iresnet/configs/v1_ir101.yaml \
models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
data_augs=configs/basic_v1.yaml \
classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.40 \
losses=configs/adaface.yaml \
evaluations=configs/skip_eval.yaml \
dataset=configs/dataset_0605_train_rec.yaml \
dataset.model_save_dir=/data1/dataset_0605/train_output"

LIMIT="trainers.limit_num_batch=1000"
OPT="models.freeze=True pefts=configs/freeze.yaml trainers.batch_size=512 \
optims=configs/step_sgd.yaml \
optims.lr=0.008 optims.num_epoch=1 \
optims.momentum=0.9 optims.weight_decay=0.0001 \
optims.max_grad_norm=5.0 \
trainers.skip_final_eval=True"

LOG_DIR="scripts/benchmark/results"
mkdir -p "$LOG_DIR"

# 监控函数
start_monitor() {
    local tag=$1
    # CPU + 内存 (每2秒)
    vmstat 2 > "$LOG_DIR/${tag}_vmstat.log" &
    VMSTAT_PID=$!
    # GPU (每2秒)
    nvidia-smi --query-gpu=timestamp,gpu_bus_id,utilization.gpu,utilization.memory,memory.used \
        --format=csv -l 2 > "$LOG_DIR/${tag}_gpu.csv" &
    NVSMI_PID=$!
    # 记录 RSS 峰值
    echo "tag=$tag" > "$LOG_DIR/${tag}_summary.txt"
}

stop_monitor() {
    local tag=$1
    kill $VMSTAT_PID 2>/dev/null || true
    kill $NVSMI_PID 2>/dev/null || true
    wait $VMSTAT_PID 2>/dev/null || true
    wait $NVSMI_PID 2>/dev/null || true

    # 提取平均 GPU 利用率
    avg_gpu=$(tail -n +2 "$LOG_DIR/${tag}_gpu.csv" | awk -F',' '{sum+=$3; n++} END{if(n>0) printf "%.1f", sum/n}')
    echo "avg_gpu_util=${avg_gpu}%" >> "$LOG_DIR/${tag}_summary.txt"

    # 峰值内存 (from vmstat: free column)
    echo "vmstat log: $LOG_DIR/${tag}_vmstat.log" >> "$LOG_DIR/${tag}_summary.txt"
}

echo "========================================"
echo "Benchmark A: PIL 解码 (当前路径)"
echo "========================================"
export DECODE_BACKEND=pil
start_monitor "pil"
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=bench_decode_pil \
    $COMMON $OPT $LIMIT 2>&1 | tee "$LOG_DIR/pil_train.log"
stop_monitor "pil"

echo ""
echo "========================================"
echo "Benchmark B: TurboJPEG 解码"
echo "========================================"
export DECODE_BACKEND=turbojpeg
start_monitor "turbojpeg"
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=bench_decode_turbojpeg \
    $COMMON $OPT $LIMIT 2>&1 | tee "$LOG_DIR/turbojpeg_train.log"
stop_monitor "turbojpeg"

echo ""
echo "========================================"
echo "Benchmark C: torchvision.io 解码"
echo "========================================"
export DECODE_BACKEND=torchvision
start_monitor "torchvision"
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=bench_decode_tvio \
    $COMMON $OPT $LIMIT 2>&1 | tee "$LOG_DIR/torchvision_train.log"
stop_monitor "torchvision"

echo ""
echo "========================================"
echo "结果汇总"
echo "========================================"
for tag in pil turbojpeg torchvision; do
    echo "--- $tag ---"
    cat "$LOG_DIR/${tag}_summary.txt"
    # 从训练 log 提取 Speed 和 Epoch Time
    grep -o "Speed [0-9]*" "$LOG_DIR/${tag}_train.log" | tail -1 || true
    grep "Epoch Time" "$LOG_DIR/${tag}_train.log" | tail -1 || true
    echo ""
done

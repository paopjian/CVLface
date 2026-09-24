#!/bin/bash
# bench_augmenter_v2.sh — 对比 v1 vs v2 augmenter 在 7卡 DDP 实际训练中的速度
#
# 公共条件:
#   - 7 GPU DDP, batch_size=512, backbone 冻结
#   - RecordIO + TurboJPEG 解码 (DECODE_BACKEND=turbojpeg)
#   - 1000 batch (512,000 张/组)
#   - persistent_workers=True, prefetch_factor=3

set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg

COMMON="trainers.num_gpu=7 trainers.num_workers=8 \
trainers.precision=bf16-mixed trainers.batch_size=512 \
trainers.limit_num_batch=1000 \
models=iresnet/configs/v1_ir101.yaml \
models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
dataset=configs/dataset_0605_train_rec.yaml \
classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.40 \
losses=configs/adaface.yaml \
evaluations=configs/skip_eval.yaml \
dataset.model_save_dir=/data1/dataset_0605/train_output \
pefts=configs/freeze.yaml \
models.freeze=True \
optims=configs/step_sgd.yaml \
optims.lr=0.008 optims.num_epoch=1 \
optims.momentum=0.9 optims.weight_decay=0.0001 \
optims.max_grad_norm=5.0 \
trainers.skip_final_eval=True"

LOG_DIR="scripts/benchmark/results"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "[1/5] Baseline: basic_v1 (PIL-based)"
echo "=========================================="
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py $COMMON data_augs=configs/basic_v1.yaml \
    trainers.prefix=bench_aug_basic_v1 2>&1 | tee "$LOG_DIR/aug_basic_v1.log"

echo ""
echo "=========================================="
echo "[2/5] V2 Numpy: basic_v2_numpy"
echo "=========================================="
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py $COMMON data_augs=configs/basic_v2_numpy.yaml \
    trainers.prefix=bench_aug_basic_v2_numpy 2>&1 | tee "$LOG_DIR/aug_basic_v2_numpy.log"

echo ""
echo "=========================================="
echo "[3/5] Baseline: gridsample_v1 (PIL-mixed)"
echo "=========================================="
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py $COMMON data_augs=configs/gridsample_v1.yaml \
    trainers.prefix=bench_aug_gridsample_v1 2>&1 | tee "$LOG_DIR/aug_gridsample_v1.log"

echo ""
echo "=========================================="
echo "[4/5] V2 Numpy: gridsample_v2_numpy"
echo "=========================================="
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py $COMMON data_augs=configs/gridsample_v2_numpy.yaml \
    trainers.prefix=bench_aug_gridsample_v2_numpy 2>&1 | tee "$LOG_DIR/aug_gridsample_v2_numpy.log"

echo ""
echo "=========================================="
echo "[5/5] V2 GPU: gridsample_v2_gpu"
echo "=========================================="
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py $COMMON data_augs=configs/gridsample_v2_gpu.yaml \
    trainers.prefix=bench_aug_gridsample_v2_gpu 2>&1 | tee "$LOG_DIR/aug_gridsample_v2_gpu.log"

echo ""
echo "=========================================="
echo "结果汇总"
echo "=========================================="
for tag in aug_basic_v1 aug_basic_v2_numpy aug_gridsample_v1 aug_gridsample_v2_numpy aug_gridsample_v2_gpu; do
    echo "--- $tag ---"
    grep "Epoch Time" "$LOG_DIR/${tag}.log" | tail -1 || true
    echo ""
done

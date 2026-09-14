#!/bin/bash
# 快速验证脚本 - 测试单个 epoch 的评估是否能正常运行

set -e

echo "================================================"
echo "测试 TRT 评估脚本"
echo "================================================"

# 测试 checkpoint 路径
CKPT_PATH="/data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch/epoch:0"

if [ ! -d "$CKPT_PATH" ]; then
    echo "错误: checkpoint 路径不存在: $CKPT_PATH"
    exit 1
fi

echo ""
echo "检查 checkpoint 结构..."
ls -l "$CKPT_PATH" | grep -E "(model\.(yaml|pt)|config\.yaml)"

echo ""
echo "检查模型配置..."
cat "$CKPT_PATH/model.yaml"

echo ""
echo "================================================"
echo "运行单 GPU 测试评估（仅第一个测试集）"
echo "================================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/root/miniconda3/envs/cvlface/lib:${LD_LIBRARY_PATH}"

# 运行评估（使用 timeout 防止卡死）
timeout 600 python eval_all_trt_single.py \
  --num_gpu 1 \
  --eval_config_name test_20260605 \
  --ckpt_path "$CKPT_PATH" \
  --name test_verification \
  --precision fp16

echo ""
echo "================================================"
echo "检查输出结果"
echo "================================================"

RESULT_DIR="eval3_results/test_verification"
if [ -d "$RESULT_DIR" ]; then
    echo "结果目录内容:"
    ls -lh "$RESULT_DIR"

    if [ -f "$RESULT_DIR/epoch_0_summary.csv" ]; then
        echo ""
        echo "Summary 结果:"
        cat "$RESULT_DIR/epoch_0_summary.csv"
    fi
else
    echo "警告: 结果目录不存在"
fi

echo ""
echo "测试完成!"

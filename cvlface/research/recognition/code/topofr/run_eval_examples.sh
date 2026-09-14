#!/bin/bash
# 完整的评估示例脚本

echo "=========================================="
echo "TopoFR Checkpoint 评估脚本"
echo "=========================================="

# 配置参数
NUM_GPU=8
EVAL_CONFIG="test_20260605"
PROJECT_NAME="work_0605_test"
PRECISION="fp16"  # 可选: fp16 或 fp32

# Checkpoint 1: s2_topofr_body36_0605_09-03_0
echo ""
echo "准备评估: s2_topofr_body36_0605_09-03_0"
CKPT_DIR_1="/data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch"
NAME_1="s2_topofr_body36_0605_09-03_0"

if [ -d "$CKPT_DIR_1" ]; then
    echo "找到 checkpoint 目录: $CKPT_DIR_1"
    echo "Epoch 数量: $(ls -1 $CKPT_DIR_1 | wc -l)"

    echo ""
    echo "运行命令:"
    echo "python eval_all_trt_launcher.py \\"
    echo "  --num_gpu $NUM_GPU \\"
    echo "  --eval_config_name $EVAL_CONFIG \\"
    echo "  --ckpt_dir $CKPT_DIR_1 \\"
    echo "  --project_name $PROJECT_NAME \\"
    echo "  --name $NAME_1 \\"
    echo "  --precision $PRECISION"

    # 取消注释以执行
    # python eval_all_trt_launcher.py \
    #   --num_gpu $NUM_GPU \
    #   --eval_config_name $EVAL_CONFIG \
    #   --ckpt_dir $CKPT_DIR_1 \
    #   --project_name $PROJECT_NAME \
    #   --name $NAME_1 \
    #   --precision $PRECISION
else
    echo "错误: 未找到 checkpoint 目录: $CKPT_DIR_1"
fi

echo ""
echo "=========================================="

# Checkpoint 2: s4_topofr_full_0605_09-04_0
echo ""
echo "准备评估: s4_topofr_full_0605_09-04_0"
CKPT_DIR_2="/data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch"
NAME_2="s4_topofr_full_0605_09-04_0"

if [ -d "$CKPT_DIR_2" ]; then
    echo "找到 checkpoint 目录: $CKPT_DIR_2"
    echo "Epoch 数量: $(ls -1 $CKPT_DIR_2 | wc -l)"

    echo ""
    echo "运行命令:"
    echo "python eval_all_trt_launcher.py \\"
    echo "  --num_gpu $NUM_GPU \\"
    echo "  --eval_config_name $EVAL_CONFIG \\"
    echo "  --ckpt_dir $CKPT_DIR_2 \\"
    echo "  --project_name $PROJECT_NAME \\"
    echo "  --name $NAME_2 \\"
    echo "  --precision $PRECISION"

    # 取消注释以执行
    # python eval_all_trt_launcher.py \
    #   --num_gpu $NUM_GPU \
    #   --eval_config_name $EVAL_CONFIG \
    #   --ckpt_dir $CKPT_DIR_2 \
    #   --project_name $PROJECT_NAME \
    #   --name $NAME_2 \
    #   --precision $PRECISION
else
    echo "错误: 未找到 checkpoint 目录: $CKPT_DIR_2"
fi

echo ""
echo "=========================================="
echo "说明:"
echo "1. 上述命令当前不会执行（已注释）"
echo "2. 取消脚本中的注释以实际运行评估"
echo "3. 或者直接复制命令到终端执行"
echo "4. 结果将保存到 eval3_results/<name>/ 目录"
echo "5. 同时会上传到 wandb 项目: $PROJECT_NAME"
echo "=========================================="

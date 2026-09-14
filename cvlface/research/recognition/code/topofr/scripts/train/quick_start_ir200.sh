#!/usr/bin/env bash
################################################################################
# IR-200 TopoFR 四阶段训练 - 快速启动脚本
#
# 使用方法:
#   bash scripts/train/quick_start_ir200.sh
#
# 或自定义参数:
#   DATA_ROOT=/custom/path OUTPUT_ROOT=/custom/output bash scripts/train/quick_start_ir200.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPOFR_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================="
echo "IR-200 TopoFR 四阶段训练 - 快速启动"
echo "========================================="
echo ""
echo "工作目录: $TOPOFR_DIR"
echo "训练脚本: $SCRIPT_DIR/train_topofr_ir200_0605.sh"
echo ""

# 检查预训练模型
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200/Glint360K_R200_TopoFR_9784.pt}"
if [[ ! -f "$PRETRAINED_MODEL" ]]; then
    echo "❌ 错误: 预训练模型不存在"
    echo "   路径: $PRETRAINED_MODEL"
    echo ""
    echo "请检查预训练模型路径或通过环境变量指定:"
    echo "   PRETRAINED_MODEL=/path/to/model.pt bash $0"
    exit 1
fi
echo "✓ 预训练模型: $PRETRAINED_MODEL"
echo "  大小: $(du -h "$PRETRAINED_MODEL" | cut -f1)"

# 检查数据目录
DATA_ROOT="${DATA_ROOT:-/data1/dataset_0605}"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "❌ 错误: 数据目录不存在"
    echo "   路径: $DATA_ROOT"
    echo ""
    echo "请通过环境变量指定正确的数据目录:"
    echo "   DATA_ROOT=/path/to/data bash $0"
    exit 1
fi
echo "✓ 数据目录: $DATA_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/train_output}"
echo "✓ 输出目录: $OUTPUT_ROOT"

# 显示训练配置
echo ""
echo "训练配置:"
echo "  - 模型: IR-200 (6-26-60-6 blocks)"
echo "  - GPU: 8 卡"
echo "  - Batch Size: 128 per GPU (总计 1024)"
echo "  - Precision: bf16-mixed"
echo "  - 分类器: PartialFC (sample_rate=0.40)"
echo "  - 损失函数: TopoFR + AdaFace"
echo ""
echo "四阶段训练:"
echo "  阶段1: PartialFC warmup (5 epochs, backbone冻结)"
echo "  阶段2: layer3.48解冻 (15 epochs, classifier冻结)"
echo "  阶段3: PartialFC重训练 (5 epochs, backbone冻结)"
echo "  阶段4: 全模型微调 (15 epochs, classifier冻结)"
echo ""

# 询问确认
read -p "是否开始训练? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消训练"
    exit 0
fi

echo ""
echo "========================================="
echo "开始训练..."
echo "========================================="
echo ""

# 导出环境变量并执行训练脚本
export DATA_ROOT
export OUTPUT_ROOT
export PRETRAINED_MODEL

cd "$TOPOFR_DIR"
exec bash "$SCRIPT_DIR/train_topofr_ir200_0605.sh"

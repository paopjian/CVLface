#!/bin/bash
################################################################################
# BN 冻结对比实验 - 自动化运行脚本
################################################################################
# 实验目标:
#   验证 BN 层冻结对预训练模型微调的影响
#
# 实验设计:
#   1. 冻结 BN: 训练 0%/10%/20%/30%/40%/50% 时评估 agedb_30
#   2. 不冻结 BN: 训练 0%/10%/20%/30%/40%/50% 时评估 agedb_30
#   3. 对比两组结果
#
# 预期结果:
#   - 冻结 BN: 预训练性能保持稳定 (~98%)
#   - 不冻结 BN: 第一个训练阶段后性能下降 (如你观察到的 72%)
################################################################################

set -e

echo "================================================================================"
echo "BN 冻结对比实验"
echo "================================================================================"
echo ""
echo "实验配置:"
echo "  - 预训练模型: Glint360K_R100_TopoFR"
echo "  - 数据集: dataset_0605 (791,509 类)"
echo "  - 评估基准: agedb_30"
echo "  - 评估节点: 0%, 10%, 20%, 30%, 40%, 50% 训练进度"
echo "  - GPU: 8x (CUDA_VISIBLE_DEVICES=0-7)"
echo ""
echo "================================================================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /root/anaconda3/bin/activate cvlface
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

# 实验 1: 冻结 BN
echo ""
echo "================================================================================"
echo "[实验 1/2] 冻结 BN 层"
echo "================================================================================"
echo "预期: 预训练性能保持稳定，各阶段 agedb_30 维持在 ~98%"
echo ""

set +e
fabric run --devices=8 --precision="bf16-mixed" \
    train_bn_freeze_experiment.py \
    freeze_bn=True

FROZEN_EXIT_CODE=$?
set -e

if [ $FROZEN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ 冻结 BN 实验失败，退出码: ${FROZEN_EXIT_CODE}"
    echo "跳过后续实验"
    exit $FROZEN_EXIT_CODE
fi

echo ""
echo "✓ 冻结 BN 实验完成"
echo ""

# 等待 5 秒，让 GPU 冷却
sleep 5

# 实验 2: 不冻结 BN
echo ""
echo "================================================================================"
echo "[实验 2/2] 不冻结 BN 层 (标准微调)"
echo "================================================================================"
echo "预期: BN 统计量变化导致性能下降，第一阶段后 agedb_30 降至 ~72%"
echo ""

set +e
fabric run --devices=8 --precision="bf16-mixed" \
    train_bn_freeze_experiment.py \
    freeze_bn=False

UNFROZEN_EXIT_CODE=$?
set -e

if [ $UNFROZEN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ 不冻结 BN 实验失败，退出码: ${UNFROZEN_EXIT_CODE}"
    exit $UNFROZEN_EXIT_CODE
fi

echo ""
echo "✓ 不冻结 BN 实验完成"
echo ""

# 汇总结果
echo "================================================================================"
echo "实验完成！生成对比报告..."
echo "================================================================================"

python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

# 查找结果文件
base_dir = Path("/data1/dataset_0605/train_output")
frozen_file = None
unfrozen_file = None

for exp_dir in sorted(base_dir.glob("bn_freeze_exp_*"), reverse=True):
    if "frozen" in exp_dir.name and frozen_file is None:
        candidate = exp_dir / "bn_freeze_results_frozen.json"
        if candidate.exists():
            frozen_file = candidate
    elif "unfrozen" in exp_dir.name and unfrozen_file is None:
        candidate = exp_dir / "bn_freeze_results_unfrozen.json"
        if candidate.exists():
            unfrozen_file = candidate
    if frozen_file and unfrozen_file:
        break

if not frozen_file or not unfrozen_file:
    print("错误: 未找到完整的实验结果文件")
    print(f"  冻结 BN: {frozen_file}")
    print(f"  不冻结 BN: {unfrozen_file}")
    exit(1)

# 读取结果
with open(frozen_file) as f:
    frozen_data = json.load(f)
with open(unfrozen_file) as f:
    unfrozen_data = json.load(f)

frozen_results = frozen_data['results']
unfrozen_results = unfrozen_data['results']

# 打印对比报告
print("\n" + "=" * 80)
print("BN 冻结对比实验 - 结果报告")
print("=" * 80)
print()
print("agedb_30 准确率对比:")
print("-" * 80)
print(f"{'训练进度':<10} {'冻结 BN':>15} {'不冻结 BN':>15} {'差异':>15}")
print("-" * 80)

for progress in ['0%', '10%', '20%', '30%', '40%', '50%']:
    frozen_acc = frozen_results.get(progress, 0.0) * 100
    unfrozen_acc = unfrozen_results.get(progress, 0.0) * 100
    diff = frozen_acc - unfrozen_acc
    diff_sign = "+" if diff > 0 else ""
    print(f"{progress:<10} {frozen_acc:>14.2f}% {unfrozen_acc:>14.2f}% {diff_sign}{diff:>13.2f}%")

print("-" * 80)
print()

# 关键发现
print("关键发现:")
print("-" * 80)

# 预训练模型性能
pretrain_frozen = frozen_results.get('0%', 0.0) * 100
pretrain_unfrozen = unfrozen_results.get('0%', 0.0) * 100
print(f"1. 预训练模型基准: {pretrain_frozen:.2f}% (两组应相同)")

# 第一阶段下降
if '10%' in frozen_results and '10%' in unfrozen_results:
    drop_frozen = pretrain_frozen - frozen_results['10%'] * 100
    drop_unfrozen = pretrain_unfrozen - unfrozen_results['10%'] * 100
    print(f"2. 训练至 10% 后性能下降:")
    print(f"   - 冻结 BN:   {drop_frozen:>6.2f}% (应该很小)")
    print(f"   - 不冻结 BN: {drop_unfrozen:>6.2f}% (应该较大)")

# 最终性能
final_frozen = frozen_results.get('50%', 0.0) * 100
final_unfrozen = unfrozen_results.get('50%', 0.0) * 100
print(f"3. 训练至 50% 后性能:")
print(f"   - 冻结 BN:   {final_frozen:.2f}%")
print(f"   - 不冻结 BN: {final_unfrozen:.2f}%")

print("-" * 80)
print()

# 结论
print("结论:")
print("-" * 80)
if drop_frozen < 3 and drop_unfrozen > 15:
    print("✓ 实验验证了假设:")
    print("  - 冻结 BN 可以保护预训练特征，避免早期性能崩溃")
    print("  - 不冻结 BN 导致特征分布剧烈变化，需要重新学习")
    print()
    print("建议:")
    print("  1. 微调预训练模型时，考虑冻结 BN 层")
    print("  2. 或使用更小的学习率 + 更长的 warmup")
    print("  3. 或采用渐进式解冻策略")
else:
    print("⚠ 结果与预期不完全一致，可能需要:")
    print("  - 检查学习率设置")
    print("  - 增加训练步数")
    print("  - 验证 BN 冻结实现")

print("-" * 80)
print()
print(f"详细结果:")
print(f"  冻结 BN:   {frozen_file}")
print(f"  不冻结 BN: {unfrozen_file}")
print("=" * 80)
PYTHON_SCRIPT

echo ""
echo "================================================================================"
echo "所有实验完成！"
echo "================================================================================"

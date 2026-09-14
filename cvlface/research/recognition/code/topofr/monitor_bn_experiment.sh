#!/bin/bash
################################################################################
# BN 冻结实验监控脚本
################################################################################

LOG_FILE="/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/bn_experiment.log"

echo "================================================================================"
echo "BN 冻结实验监控"
echo "================================================================================"
echo ""

# 检查进程
echo "进程状态:"
echo "------------------------------------------------------------------------"
PIDS=$(ps aux | grep -E "(train_bn_freeze|fabric run.*bn_freeze)" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
if [ -z "$PIDS" ]; then
    echo "✗ 未发现运行中的实验进程"
else
    echo "✓ 实验进程运行中 (PIDs: $PIDS)"
fi
echo ""

# 检查日志
if [ ! -f "$LOG_FILE" ]; then
    echo "✗ 日志文件不存在: $LOG_FILE"
    exit 1
fi

echo "最新日志 (最后 50 行):"
echo "------------------------------------------------------------------------"
tail -50 "$LOG_FILE"
echo ""

# 提取关键信息
echo "================================================================================"
echo "实验进度摘要"
echo "================================================================================"
echo ""

# 检查哪个实验在运行
if grep -q "冻结 BN 层" "$LOG_FILE" 2>/dev/null; then
    CURRENT_EXP=$(grep -E "^\[实验 [0-9]/2\]" "$LOG_FILE" | tail -1)
    echo "当前阶段: $CURRENT_EXP"
fi

# 提取评估结果
if grep -q "agedb_30" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "已完成的评估:"
    echo "------------------------------------------------------------------------"
    grep -E "(预训练模型|[0-9]+% 训练):.*agedb_30" "$LOG_FILE" | tail -10
fi

# 检查是否有错误
ERROR_COUNT=$(grep -i "error\|traceback\|exception" "$LOG_FILE" 2>/dev/null | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "⚠ 发现 $ERROR_COUNT 个错误信息，检查日志:"
    grep -i "error\|exception" "$LOG_FILE" | tail -5
fi

echo ""
echo "================================================================================"
echo "使用以下命令持续监控:"
echo "  watch -n 10 bash $0"
echo ""
echo "查看完整日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "手动停止实验:"
echo "  kill $PIDS"
echo "================================================================================"

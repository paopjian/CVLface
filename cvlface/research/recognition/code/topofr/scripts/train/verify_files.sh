#!/bin/bash
echo "================================================================================"
echo "TopoFR 集成文件验证"
echo "================================================================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

success=0
total=0

echo "新增文件检查:"
echo "--------------------------------------------------------------------------------"
files=(
    "losses/topology.py"
    "losses/gum.py"
    "losses/topofr_loss.py"
    "losses/configs/topofr.yaml"
    "losses/configs/topofr_cosface.yaml"
    "losses/configs/topofr_adaface.yaml"
    "pipelines/train_model_cls_topofr_pipeline.py"
    "test_topofr_integration.py"
    "TOPOFR_README.md"
    "TOPOFR_CHANGES.md"
    "INTEGRATION_SUMMARY.md"
    "CHECKLIST.md"
    "train_topofr_0605.sh"
    "README_TOPOFR.txt"
)

for file in "${files[@]}"; do
    total=$((total + 1))
    if check_file "$file"; then
        success=$((success + 1))
    fi
done

echo ""
echo "修改文件检查 (应该存在):"
echo "--------------------------------------------------------------------------------"
modified_files=(
    "losses/__init__.py"
    "classifiers/partial_fc/partial_fc.py"
    "classifiers/partial_fc/__init__.py"
    "classifiers/fc/fc.py"
    "pipelines/__init__.py"
)

for file in "${modified_files[@]}"; do
    total=$((total + 1))
    if check_file "$file"; then
        success=$((success + 1))
    fi
done

echo ""
echo "================================================================================"
echo "验证结果: $success/$total 文件存在"
echo "================================================================================"

if [ $success -eq $total ]; then
    echo -e "${GREEN}✓ 所有文件验证通过！${NC}"
    exit 0
else
    echo -e "${RED}✗ 部分文件缺失，请检查！${NC}"
    exit 1
fi

#!/usr/bin/env bash
# 测试 IR-200 模型加载和配置

set -euo pipefail

export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}"
source "/root/anaconda3/bin/activate" cvlface
cd "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr"

echo "========================================="
echo "测试 IR-200 TopoFR 配置"
echo "========================================="

PRETRAINED_MODEL="/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200/Glint360K_R200_TopoFR_9784.pt"

python -c "
import sys
import torch
from pathlib import Path

# 加载预训练模型
print('1. 检查预训练模型文件...')
model_path = Path('$PRETRAINED_MODEL')
if not model_path.exists():
    print(f'❌ 预训练模型不存在: {model_path}')
    sys.exit(1)
print(f'✓ 预训练模型存在: {model_path}')
print(f'  大小: {model_path.stat().st_size / (1024**3):.2f} GB')

# 加载权重
print('\n2. 加载预训练权重...')
ckpt = torch.load(model_path, map_location='cpu')
print(f'✓ 成功加载，共 {len(ckpt)} 个参数')

# 分析层结构
print('\n3. 分析模型结构...')
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    blocks = set()
    for k in ckpt.keys():
        if k.startswith(layer_name + '.'):
            block_idx = k.split('.')[1]
            if block_idx.isdigit():
                blocks.add(int(block_idx))
    if blocks:
        max_block = max(blocks)
        print(f'  {layer_name}: {max_block + 1} blocks')

print('\n4. 测试模型初始化...')
from models.iresnet_insightface.model import iresnet200
net = iresnet200()
print(f'✓ IR-200 模型初始化成功')

# 测试加载预训练权重
print('\n5. 测试加载预训练权重到模型...')
result = net.load_state_dict(ckpt, strict=False)
if result.missing_keys:
    print(f'  缺失的键: {result.missing_keys[:5]}...' if len(result.missing_keys) > 5 else f'  缺失的键: {result.missing_keys}')
if result.unexpected_keys:
    print(f'  意外的键: {result.unexpected_keys[:5]}...' if len(result.unexpected_keys) > 5 else f'  意外的键: {result.unexpected_keys}')
if not result.missing_keys and not result.unexpected_keys:
    print('✓ 权重完全匹配')
else:
    print('⚠ 权重部分匹配（可能正常，检查 missing/unexpected keys）')

# 测试前向传播
print('\n6. 测试前向传播...')
net.eval()
with torch.no_grad():
    x = torch.randn(2, 3, 112, 112)
    out = net(x)
    print(f'✓ 前向传播成功')
    print(f'  输入: {x.shape}')
    print(f'  输出: {out.shape}')

print('\n========================================')
print('✓ IR-200 配置测试通过')
print('========================================')
"

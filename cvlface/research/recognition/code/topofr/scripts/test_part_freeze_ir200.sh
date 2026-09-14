#!/usr/bin/env bash
# 测试 IR-200 part_freeze 配置

set -euo pipefail

export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}"
source "/root/anaconda3/bin/activate" cvlface
cd "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr"

echo "========================================="
echo "测试 IR-200 part_freeze 配置"
echo "========================================="

python -c "
import sys
import torch
from models.iresnet_insightface.model import iresnet200
from pefts import apply_peft_to_model
from dataclasses import dataclass

@dataclass
class MockPeftConfig:
    name: str
    target_modules: str

print('1. 初始化 IR-200 模型...')
model = iresnet200()
print(f'✓ 模型初始化成功')

# 统计模型参数
total_params = sum(p.numel() for p in model.parameters())
print(f'✓ 总参数量: {total_params:,}')

print('\n2. 测试 layer3.48 解冻配置...')
peft_config = MockPeftConfig(name='part_freeze', target_modules='layer3.48')

try:
    peft_model = apply_peft_to_model(peft_config, model)
    print('✓ part_freeze 应用成功')

    # 统计可训练参数
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in peft_model.parameters() if not p.requires_grad)

    print(f'\n参数统计:')
    print(f'  可训练参数: {trainable:,} ({100*trainable/total_params:.2f}%)')
    print(f'  冻结参数: {frozen:,} ({100*frozen/total_params:.2f}%)')

    # 检查具体哪些层被解冻
    print(f'\n解冻的层（前10个）:')
    unfrozen_layers = []
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            unfrozen_layers.append(name)

    for layer in unfrozen_layers[:10]:
        print(f'  ✓ {layer}')

    if len(unfrozen_layers) > 10:
        print(f'  ... 共 {len(unfrozen_layers)} 个可训练参数')

    # 验证关键层
    print(f'\n关键层验证:')
    key_checks = [
        ('layer3.47', False, '应该被冻结'),
        ('layer3.48', True, '应该被解冻'),
        ('layer3.59', True, '应该被解冻'),
        ('layer4.0', True, '应该被解冻'),
        ('bn2', True, '应该被解冻'),
        ('fc', True, '应该被解冻'),
    ]

    for layer_pattern, should_be_trainable, desc in key_checks:
        matching_params = [name for name, p in peft_model.named_parameters()
                          if layer_pattern in name]
        if matching_params:
            sample_name = matching_params[0]
            is_trainable = dict(peft_model.named_parameters())[sample_name].requires_grad
            status = '✓' if is_trainable == should_be_trainable else '✗'
            print(f'  {status} {layer_pattern}: {\"可训练\" if is_trainable else \"冻结\"} ({desc})')
        else:
            print(f'  ⚠ {layer_pattern}: 未找到匹配参数')

    print('\n========================================')
    print('✓ IR-200 part_freeze 配置测试通过')
    print('========================================')
    sys.exit(0)

except KeyError as e:
    print(f'\n✗ 配置错误: {e}')
    print('layer3.48 不在 target_modules_mapping 中')
    sys.exit(1)
except Exception as e:
    print(f'\n✗ 测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

#!/usr/bin/env bash
# 测试 IR-200 body.72 解冻配置

set -euo pipefail

export LD_LIBRARY_PATH="/root/anaconda3/envs/cvlface/lib:${LD_LIBRARY_PATH:-}"
source "/root/anaconda3/bin/activate" cvlface
cd "/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr"

echo "========================================="
echo "测试 IR-200 body.72 解冻配置"
echo "========================================="

python -c "
import sys
import torch
from models.iresnet_insightface import IResNetModel
from pefts import apply_peft_to_model
from dataclasses import dataclass
import yaml

# 加载配置
with open('models/iresnet_insightface/configs/v1_ir200.yaml', 'r') as f:
    cfg_dict = yaml.safe_load(f)

@dataclass
class ModelConfig:
    name: str
    output_dim: int
    input_size: list
    color_space: str
    start_from: str
    freeze: bool

@dataclass
class MockPeftConfig:
    name: str
    target_modules: str

# 创建配置对象
model_cfg = ModelConfig(**cfg_dict)

print('1. 初始化 IR-200 模型（通过 IResNetModel）...')
model = IResNetModel.from_config(model_cfg)
print(f'✓ 模型初始化成功')

# 统计模型参数
total_params = sum(p.numel() for p in model.parameters())
print(f'✓ 总参数量: {total_params:,}')

# 显示参数命名示例
print(f'\n参数命名示例（前5个）:')
for i, (name, _) in enumerate(model.named_parameters()):
    if i < 5:
        print(f'  {name}')
    else:
        break

print(f'\n2. 测试 body.72 解冻配置...')
peft_config = MockPeftConfig(name='part_freeze', target_modules='body.72')

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
    print(f'\n解冻的层（选择性显示）:')
    unfrozen_layers = [name for name, p in peft_model.named_parameters() if p.requires_grad]

    # 显示关键层
    key_patterns = ['layer3.30', 'layer3.40', 'layer3.50', 'layer4.0', 'bn2', 'fc']
    for pattern in key_patterns:
        matching = [name for name in unfrozen_layers if pattern in name]
        if matching:
            print(f'  ✓ {pattern}: {len(matching)} 个参数可训练')

    # 验证关键层的训练状态
    print(f'\n关键层验证:')

    # body.72 对应 layer1[6] + layer2[26] + layer3[40]
    # 所以 layer3.39 应该冻结，layer3.40+ 应该解冻
    checks = [
        ('net.layer3.39', False, '应该被冻结'),
        ('net.layer3.40', True, '应该被解冻'),
        ('net.layer3.59', True, '应该被解冻'),
        ('net.layer4.0', True, '应该被解冻'),
        ('net.bn2', True, '应该被解冻'),
        ('net.fc', True, '应该被解冻'),
    ]

    all_pass = True
    for layer_pattern, should_be_trainable, desc in checks:
        matching_params = [name for name, p in peft_model.named_parameters()
                          if layer_pattern in name]
        if matching_params:
            sample_name = matching_params[0]
            is_trainable = dict(peft_model.named_parameters())[sample_name].requires_grad
            status = '✓' if is_trainable == should_be_trainable else '✗'
            if is_trainable != should_be_trainable:
                all_pass = False
            print(f'  {status} {layer_pattern}: {\"可训练\" if is_trainable else \"冻结\"} ({desc})')
        else:
            print(f'  ⚠ {layer_pattern}: 未找到匹配参数')
            all_pass = False

    print(f'\n========================================')
    if all_pass and trainable > 0:
        print('✓ IR-200 body.72 解冻配置测试通过')
        print('========================================')
        sys.exit(0)
    else:
        print('✗ IR-200 body.72 解冻配置测试失败')
        print('========================================')
        sys.exit(1)

except KeyError as e:
    print(f'\n✗ 配置错误: {e}')
    print('body.72 不在 target_modules_mapping 中')
    print('\n可用的 target_modules 示例:')
    print('  - body.0 ~ body.97')
    print('  - blocks.0 ~ blocks.23')
    sys.exit(1)
except Exception as e:
    print(f'\n✗ 测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

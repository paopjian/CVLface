#!/usr/bin/env python
"""测试模型加载是否正常"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
import torch
from models import get_model
from general_utils.config_utils import load_config

# 测试两个checkpoint
checkpoints = [
    '/data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch/epoch:0',
    '/data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch/epoch:0',
]

for ckpt_path in checkpoints:
    print(f"\n{'='*60}")
    print(f"测试: {ckpt_path}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print(f"  路径不存在，跳过")
        continue

    # 加载配置
    model_yaml = os.path.join(ckpt_path, 'model.yaml')
    model_pt = os.path.join(ckpt_path, 'model.pt')

    print(f"  加载配置: {model_yaml}")
    model_config = load_config(model_yaml)
    print(f"    yaml_path: {model_config.yaml_path}")
    print(f"    name: {model_config.name}")
    print(f"    output_dim: {model_config.output_dim}")

    # 清空 start_from 和 freeze
    model_config.start_from = ''
    model_config.freeze = False

    # 创建模型
    print(f"  创建模型...")
    model = get_model(model_config)

    # 加载权重
    print(f"  加载权重: {model_pt}")
    model.load_state_dict_from_path(model_pt)
    model.eval()

    # 测试前向传播
    print(f"  测试前向传播...")
    model = model.cuda()
    dummy_input = torch.randn(2, 3, 112, 112).cuda()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"    输入: {dummy_input.shape}")
    print(f"    输出: {output.shape}")

    print(f"  ✓ 模型加载成功!")

    del model
    torch.cuda.empty_cache()

print("\n所有测试完成!")

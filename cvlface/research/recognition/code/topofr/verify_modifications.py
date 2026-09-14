#!/usr/bin/env python
"""
验证脚本修改的完整性测试
"""
import os
import sys

print("="*60)
print("验证 TRT 评估脚本修改")
print("="*60)

# 1. 检查文件是否存在
files_to_check = [
    'eval_all_trt_single.py',
    'eval_all_trt_launcher.py',
    'test_model_load.py',
    'EVAL_TRT_USAGE.md',
    'test_eval_trt.sh',
]

print("\n1. 检查文件存在性:")
for f in files_to_check:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")

# 2. 检查 checkpoint 路径
print("\n2. 检查 checkpoint 路径:")
ckpt_dirs = [
    '/data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch',
    '/data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch',
]

for d in ckpt_dirs:
    exists = os.path.exists(d)
    status = "✓" if exists else "✗"
    print(f"  {status} {d}")
    if exists:
        epochs = [e for e in os.listdir(d) if e.startswith('epoch:')]
        print(f"      共 {len(epochs)} 个 epoch")

# 3. 检查评估配置
print("\n3. 检查评估配置:")
eval_config = 'evaluations/configs/test_20260605.yaml'
exists = os.path.exists(eval_config)
status = "✓" if exists else "✗"
print(f"  {status} {eval_config}")

# 4. 检查模型定义
print("\n4. 检查模型定义:")
model_files = [
    'models/__init__.py',
    'models/iresnet_insightface/__init__.py',
    'models/iresnet_insightface/configs/v1_ir101.yaml',
]

for f in model_files:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")

# 5. 检查关键修改
print("\n5. 检查关键修改:")
with open('eval_all_trt_single.py', 'r') as f:
    content = f.read()

    # 检查是否移除了 'work_0605' 参数
    if "get_model(model_config, 'work_0605')" in content:
        print("  ✗ 仍然包含 get_model(model_config, 'work_0605')")
    elif "get_model(model_config)" in content:
        print("  ✓ 已修改为 get_model(model_config)")
    else:
        print("  ? 未找到 get_model 调用")

print("\n" + "="*60)
print("验证完成!")
print("="*60)

print("\n下一步:")
print("1. 运行 test_model_load.py 测试模型加载")
print("2. 运行 bash test_eval_trt.sh 测试单个评估")
print("3. 使用完整命令评估所有 checkpoint")

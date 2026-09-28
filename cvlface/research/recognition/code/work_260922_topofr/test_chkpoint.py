#!/usr/bin/env python3
"""
测试checkpoint加载的脚本
用于验证之前保存的checkpoint是否能正确加载
"""

import os
import sys
import torch
import pyrootutils

# 设置根路径
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(str(root))

def test_checkpoint_loading(checkpoint_path):
    """测试checkpoint是否能正确加载"""
    
    print(f"Testing checkpoint: {checkpoint_path}")
    
    # 检查checkpoint目录是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint directory does not exist: {checkpoint_path}")
        return False
    
    # 检查必要的文件是否存在
    required_files = ['model.pt', 'optimizer.pt', 'lr_scheduler.pt', 'pipeline.pt', 'config.yaml']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    # 检查分类器文件 - 可能是单个文件或分布式文件
    classifier_files = []
    if os.path.exists(os.path.join(checkpoint_path, 'classifier.pt')):
        classifier_files.append('classifier.pt')
    else:
        # 检查分布式分类器文件
        for i in range(8):  # 假设最多8个GPU
            classifier_rank_file = f'classifier_rank{i}.pt'
            classifier_rank_path = os.path.join(checkpoint_path, classifier_rank_file)
            if os.path.exists(classifier_rank_path):
                classifier_files.append(classifier_rank_file)
    
    if not classifier_files:
        missing_files.append('classifier.pt (or classifier_rank*.pt files)')
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    try:
        # 尝试加载pipeline状态
        pipeline_path = os.path.join(checkpoint_path, 'pipeline.pt')
        pipeline_state = torch.load(pipeline_path, map_location='cpu')
        
        print(f"✅ Pipeline state loaded successfully")
        print(f"   - Epoch: {pipeline_state.get('epoch', 'Unknown')}")
        print(f"   - Step: {pipeline_state.get('step', 'Unknown')}")
        print(f"   - Images seen: {pipeline_state.get('n_images_seen', 'Unknown')}")
        print(f"   - Module names: {pipeline_state.get('module_names_list', 'Unknown')}")
        
        # 尝试加载模型权重
        model_path = os.path.join(checkpoint_path, 'model.pt')
        model_state = torch.load(model_path, map_location='cpu')
        print(f"✅ Model state loaded successfully (keys: {len(model_state)})")
        
        # 尝试加载分类器权重
        if os.path.exists(os.path.join(checkpoint_path, 'classifier.pt')):
            # 单个分类器文件
            classifier_path = os.path.join(checkpoint_path, 'classifier.pt')
            classifier_state = torch.load(classifier_path, map_location='cpu')
            print(f"✅ Classifier state loaded successfully (keys: {len(classifier_state)})")
        else:
            # 分布式分类器文件
            total_classifier_parts = 0
            for classifier_file in classifier_files:
                classifier_path = os.path.join(checkpoint_path, classifier_file)
                classifier_state = torch.load(classifier_path, map_location='cpu')
                total_classifier_parts += 1
            print(f"✅ Distributed classifier state loaded successfully ({total_classifier_parts} parts: {classifier_files})")
            print(f"   - This is a Partial FC classifier distributed across {total_classifier_parts} GPUs")
        
        # 尝试加载优化器状态
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        optimizer_state = torch.load(optimizer_path, map_location='cpu')
        print(f"✅ Optimizer state loaded successfully")
        
        # 尝试加载学习率调度器状态
        lr_scheduler_path = os.path.join(checkpoint_path, 'lr_scheduler.pt')
        lr_scheduler_state = torch.load(lr_scheduler_path, map_location='cpu')
        print(f"✅ LR scheduler state loaded successfully")
        
        print(f"✅ All checkpoint files loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def find_available_checkpoints(base_output_dir):
    """查找可用的checkpoint"""
    
    print(f"Searching for checkpoints in: {base_output_dir}")
    
    if not os.path.exists(base_output_dir):
        print(f"❌ Output directory does not exist: {base_output_dir}")
        return []
    
    checkpoints = []
    
    # 查找 checkpoints/best
    best_path = os.path.join(base_output_dir, 'checkpoints', 'best')
    if os.path.exists(best_path):
        checkpoints.append(('best', best_path))
    
    # 查找 checkpoints_every_epoch
    every_epoch_dir = os.path.join(base_output_dir, 'checkpoints_every_epoch')
    if os.path.exists(every_epoch_dir):
        for item in os.listdir(every_epoch_dir):
            item_path = os.path.join(every_epoch_dir, item)
            if os.path.isdir(item_path) and item.startswith('epoch:'):
                checkpoints.append(('every_epoch', item_path))
    
    # 查找 checkpoints 目录下的其他checkpoint
    checkpoints_dir = os.path.join(base_output_dir, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        for item in os.listdir(checkpoints_dir):
            item_path = os.path.join(checkpoints_dir, item)
            if os.path.isdir(item_path) and item.startswith('epoch:'):
                checkpoints.append(('regular', item_path))
    
    return checkpoints

if __name__ == "__main__":
    # 请修改这个路径为你的实际输出目录
    base_output_dir = "/root/zhaokj/CVLface/cvlface/research/recognition/experiments/work_85/ir101_adaface_08-06_0/checkpoints/best"  # 修改为你的实际路径
    test_checkpoint_loading(base_output_dir)
    
    # if len(sys.argv) > 1:
    #     checkpoint_path = sys.argv[1]
    #     test_checkpoint_loading(checkpoint_path)
    # else:
    #     print("Usage:")
    #     print("  python test_checkpoint.py /path/to/checkpoint/dir")
    #     print("\nOr modify the script to set your base_output_dir and run:")
    #     print("  python test_checkpoint.py")
        
    #     # 取消注释下面的代码来自动查找checkpoint
    #     base_output_dir = "/root/zhaokj/CVLface/cvlface/research/recognition/experiments/work_85/ir101_adaface_08-06_0/checkpoints/best"  # 修改这里
    #     checkpoints = find_available_checkpoints(base_output_dir)
        
    #     if not checkpoints:
    #         print("No checkpoints found!")
    #     else:
    #         print(f"Found {len(checkpoints)} checkpoints:")
    #         for i, (type_name, path) in enumerate(checkpoints):
    #             print(f"  {i+1}. [{type_name}] {path}")
            
    #         print("\nTesting the first checkpoint:")
    #         test_checkpoint_loading(checkpoints[0][1])

#!/usr/bin/env python3
"""测试预训练模型在评估数据集上的原始性能"""
import sys
sys.path.insert(0, '/root/zhaokj/CVLface')

import torch
import os
from cvlface.research.recognition.code.topofr.models.iresnet_insightface import IResNetModel
from cvlface.research.recognition.code.topofr.evaluators import make_evaluator_from_config
from cvlface.research.recognition.code.topofr.pipelines import pipeline_from_name
from omegaconf import OmegaConf

# 加载配置
config_path = '/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/evaluations/configs/val_20260605.yaml'
cfg = OmegaConf.load(config_path)

# 创建模型
model_cfg = OmegaConf.create({
    'name': 'v1_ir101',
    'depth': 101,
    'drop_ratio': 0.0,
    'fp16': False,
    'color_space': 'RGB',
    'num_features': 512,
    'start_from': '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt',
    'freeze': False
})

print("创建模型...")
model = IResNet_InsightFace.from_config(model_cfg)

# 加载预训练权重
print(f"加载预训练模型: {model_cfg.start_from}")
model.load_state_dict_from_path(model_cfg.start_from)

model = model.cuda().eval()

# 创建评估pipeline
eval_pipeline = pipeline_from_name('inference', model, None)

# 只测试几个小数据集
test_datasets = ['agedb_30', 'calfw', 'cplfw']

print("\n开始评估...")
for name in test_datasets:
    if name in cfg.per_epoch_evaluations:
        info = cfg.per_epoch_evaluations[name]
        eval_data_path = os.path.join(cfg.data_root, info.path)
        eval_type = info.evaluation_type

        print(f"\n评估 {name} (类型: {eval_type})...")

        evaluator = make_evaluator_from_config(
            name=name,
            config=info,
            data_path=eval_data_path,
            evaluation_type=eval_type,
            batch_size=info.batch_size * 4,
            num_workers=8,
            transform=eval_pipeline.make_test_transform(),
        )

        result = evaluator.evaluate(eval_pipeline, epoch=0, step=0, n_images_seen=0)

        # 打印结果
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")

print("\n评估完成！")

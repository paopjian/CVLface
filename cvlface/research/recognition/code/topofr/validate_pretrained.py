"""
验证 TopoFR 预训练模型的原始性能
在 val_20260605 数据集上测试，不进行训练
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
import sys
sys.path.append(str(root))

import torch
from models import get_model
from aligners import get_aligner
from evaluations import get_evaluator_by_name, summary
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from pipelines import pipeline_from_name
from general_utils.config_utils import load_config
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset
import datetime
from lightning.fabric.strategies import DDPStrategy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str,
                       default='/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--wandb_entity', type=str, default='kejian-zhao-tsinghua-university')
    parser.add_argument('--wandb_project', type=str, default='topofr')
    parser.add_argument('--run_name', type=str, default='topofr_val')
    args = parser.parse_args()

    print("=" * 80)
    print("TopoFR 预训练模型验证")
    print("=" * 80)
    print(f"预训练模型: {args.pretrained_model}")
    print(f"GPU数量: {args.num_gpu}")
    print(f"精度: {args.precision}")
    print("=" * 80)

    # 设置 Fabric
    torch.set_float32_matmul_precision('high')
    csv_logger = CSVLogger(root_dir='/tmp/topofr_val_logs', flush_logs_every_n_steps=1)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        save_dir='/tmp/topofr_val_wandb'
    )

    ddp_strategy = DDPStrategy(timeout=datetime.timedelta(minutes=30))
    fabric = Fabric(
        precision=args.precision,
        accelerator="auto",
        strategy=ddp_strategy,
        devices=args.num_gpu,
        loggers=[csv_logger, wandb_logger],
    )

    if args.num_gpu == 1:
        fabric.launch()

    fabric.setup_dataloader_from_dataset = partial(
        setup_dataloader_from_dataset, fabric=fabric, seed=2048
    )

    # 加载模型配置
    model_config = load_config(
        os.path.join(root, 'research/recognition/code/topofr/models/iresnet_insightface/configs/v1_ir101.yaml')
    )
    model_config.start_from = args.pretrained_model
    model_config.freeze = False
    model_config.yaml_path = '/iresnet_insightface/configs/v1_ir101.yaml'

    print("\n加载模型...")
    model = get_model(model_config, task='topofr')
    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 加载 aligner
    aligner_config = load_config(
        os.path.join(root, 'research/recognition/code/topofr/aligners/configs/none.yaml')
    )
    aligner = get_aligner(aligner_config)

    # 创建推理 pipeline
    model = fabric.setup(model)  # 只 setup 模型
    pipeline = pipeline_from_name('infer_model_pipeline', model, aligner)

    # 加载评估配置
    eval_config = load_config(
        os.path.join(root, 'research/recognition/code/topofr/evaluations/configs/val_20260605.yaml')
    )

    print("\n加载评估器...")
    transform = model.module.make_test_transform() if hasattr(model, 'module') else model.make_test_transform()
    evaluators = []
    for eval_name, eval_spec in eval_config.per_epoch_evaluations.items():
        eval_path = os.path.join(eval_config.data_root, eval_spec['path'])
        evaluator = get_evaluator_by_name(
            eval_type=eval_spec['evaluation_type'],
            name=eval_name,
            eval_data_path=eval_path,
            transform=transform,
            fabric=fabric,
            batch_size=eval_spec.get('batch_size', 256),
            num_workers=eval_spec.get('num_workers', 4)
        )
        evaluators.append(evaluator)
        print(f"  - {eval_name}")

    print("\n开始评估...")
    print("=" * 80)

    # 在 10 个 "epoch" 中记录相同的分数（模拟训练过程）
    for epoch in range(10):
        print(f"\n[Epoch {epoch}] 评估中...")

        epoch_metrics = {}
        for evaluator in evaluators:
            print(f"  评估 {evaluator.name}...")
            metrics = evaluator.evaluate(pipeline)
            epoch_metrics.update(metrics)

            # 打印关键指标
            if 'acc' in metrics:
                print(f"    准确率: {metrics['acc']:.4f}")
            if 'best_threshold' in metrics:
                print(f"    最佳阈值: {metrics['best_threshold']:.4f}")

        # 记录到 WandB
        summary_metrics = summary(epoch_metrics, epoch, step=epoch, n_images_seen=0)
        fabric.log_dict(summary_metrics, step=epoch)

        # 打印汇总
        print(f"\n[Epoch {epoch}] 汇总:")
        for key, value in sorted(summary_metrics.items()):
            if 'acc' in key.lower():
                print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

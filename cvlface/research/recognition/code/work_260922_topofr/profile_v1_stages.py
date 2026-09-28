"""
V1 评估流程分阶段计时分析脚本

对 adaface_ir101_webface12m 模型，使用 val_20260605.yaml 配置，
逐评估器测量:
  1. 特征提取耗时
  2. 评估计算耗时 (custom_verification4: sim_matrix, verification: acc, tinyface: rank)

单 GPU, bf16-mixed, 不使用 torch.compile
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os, sys, time
sys.path.append(os.path.join(root))

import numpy as np
np.bool = np.bool_

import torch
torch.set_float32_matmul_precision('high')

from models import get_model
from aligners import get_aligner
from evaluations import get_evaluator_by_name
from pipelines import pipeline_from_name
from general_utils.config_utils import load_config
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset


def timed_extract(evaluator, pipeline):
    """对评估器执行特征提取并计时，返回 (collection, collection_flip, elapsed)"""
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        collection = evaluator.extract(pipeline)
        collection_flip = evaluator.extract(pipeline, flip_images=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return collection, collection_flip, elapsed


def timed_compute_verification(evaluator, collection, collection_flip):
    """verification 评估器的 compute_metric 计时"""
    start = time.time()
    result = evaluator.compute_metric(collection, collection_flip)
    elapsed = time.time() - start
    return result, elapsed


def timed_compute_tinyface(evaluator, collection, collection_flip):
    """tinyface 评估器的 compute_metric 计时"""
    start = time.time()
    result = evaluator.compute_metric(collection, collection_flip)
    elapsed = time.time() - start
    return result, elapsed


def timed_compute_custom_v4(evaluator, collection, collection_flip):
    """
    custom_verification4 评估器: 单独计时 sim_matrix 计算
    """
    import sklearn.preprocessing
    from evaluations.cluster_utils import get_sim_matrix_large_scale_v4
    from evaluations.custom_verification_evaluator import compute_tpir_from_hist

    # 合并特征 + 归一化
    embeddings = (collection['features'] + collection_flip['features']).numpy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    query_ids = collection['labels'].numpy()

    print(f"    样本数: {len(embeddings)}, 类别数: {len(np.unique(query_ids))}")

    # 计时: get_sim_matrix_large_scale_v4
    torch.cuda.synchronize()
    start = time.time()

    target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    pos_hist, neg_hist = get_sim_matrix_large_scale_v4(
        query_feats_list=embeddings,
        query_ids=query_ids,
        num_gpus=7,
        block_size=2048 * 2,
        show_progress=True,
    )
    result, thresholds = compute_tpir_from_hist(pos_hist, neg_hist, target_fars=target_fars)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"    result: {result}")
    return result, elapsed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--compile', action='store_true', help='启用 torch.compile')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['reduce-overhead', 'max-autotune', 'default'])
    args = parser.parse_args()

    ckpt_path = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'
    eval_config_path = 'evaluations/configs/val_20260605.yaml'
    precision = args.precision

    # 初始化 Fabric (多 GPU DDP)
    csv_logger_dir = os.path.join(root, 'research/recognition/experiments', 'profile_v1')
    os.makedirs(csv_logger_dir, exist_ok=True)
    csv_logger = CSVLogger(root_dir=csv_logger_dir, flush_logs_every_n_steps=1)

    fabric = Fabric(
        precision=precision,
        accelerator="auto",
        strategy="ddp",
        devices=args.num_gpu,
        loggers=[csv_logger],
    )
    if args.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=2048)
    if fabric.local_rank == 0:
        print(f"Fabric launched: {args.num_gpu} GPU, precision={precision}")

    # 加载模型
    if fabric.local_rank == 0:
        print("\n加载模型...")
    model_load_start = time.time()
    model_config = load_config(os.path.join(ckpt_path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(ckpt_path, 'model.pt'))

    # torch.compile 加速 (必须在 fabric.setup 之前)
    if args.compile:
        if fabric.local_rank == 0:
            print(f"启用 torch.compile (mode={args.compile_mode})...")
        compile_start = time.time()
        model = torch.compile(model, mode=args.compile_mode)
        if fabric.local_rank == 0:
            print(f"torch.compile 调用耗时: {time.time() - compile_start:.2f}s (实际编译在首次 forward 时)")

    model = fabric.setup(model)
    model_load_time = time.time() - model_load_start
    if fabric.local_rank == 0:
        print(f"模型加载耗时: {model_load_time:.2f}s")

    # 构建 Pipeline
    aligner_config = load_config(os.path.join(root, 'research/recognition/code/', 'run_v1', 'aligners/configs/none.yaml'))
    aligner = get_aligner(aligner_config)
    eval_pipeline = pipeline_from_name('infer_model_pipeline', model, aligner)
    eval_pipeline.integrity_check(dataset_color_space='RGB')
    transform = eval_pipeline.make_test_transform()

    # 加载评估配置
    eval_config = load_config(eval_config_path)
    if fabric.local_rank == 0:
        print(f"\n评估配置: {eval_config_path}")
        print(f"数据根目录: {eval_config.data_root}")

    # 逐评估器 Profiling
    timing_results = []

    if fabric.local_rank == 0:
        print("\n" + "=" * 70)
        print("开始分阶段计时分析 (7 GPU DDP)")
        print("=" * 70)

    total_start = time.time()

    for name, info in eval_config.per_epoch_evaluations.items():
        eval_data_path = os.path.join(eval_config.data_root, info.path)
        eval_type = info.evaluation_type
        eval_batch_size = info.batch_size
        eval_num_workers = info.num_workers

        if fabric.local_rank == 0:
            print(f"\n{'---' * 20}")
            print(f"评估器: {name} (type={eval_type})")
            print(f"数据路径: {eval_data_path}")
            print(f"{'---' * 20}")

        if not os.path.isdir(eval_data_path):
            if fabric.local_rank == 0:
                print(f"  [跳过] 数据路径不存在: {eval_data_path}")
            timing_results.append({
                'name': name, 'type': eval_type,
                'extract_time': -1, 'compute_time': -1, 'total_time': -1,
                'note': '数据不存在'
            })
            continue

        try:
            evaluator = get_evaluator_by_name(
                eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                transform=transform, fabric=fabric,
                batch_size=eval_batch_size, num_workers=eval_num_workers
            )
            evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        except Exception as e:
            if fabric.local_rank == 0:
                print(f"  [错误] 创建评估器失败: {e}")
            timing_results.append({
                'name': name, 'type': eval_type,
                'extract_time': -1, 'compute_time': -1, 'total_time': -1,
                'note': f'初始化失败: {e}'
            })
            continue

        # 阶段1: 特征提取 (所有 rank 参与)
        if fabric.local_rank == 0:
            print(f"  [阶段1] 特征提取...")
        try:
            collection, collection_flip, extract_time = timed_extract(evaluator, eval_pipeline)
            if fabric.local_rank == 0:
                print(f"  特征提取耗时: {extract_time:.2f}s")
        except Exception as e:
            if fabric.local_rank == 0:
                print(f"  [错误] 特征提取失败: {e}")
                import traceback
                traceback.print_exc()
            timing_results.append({
                'name': name, 'type': eval_type,
                'extract_time': -1, 'compute_time': -1, 'total_time': -1,
                'note': f'特征提取失败: {e}'
            })
            continue

        # 阶段2: 评估计算 (rank 0 执行，其他 rank barrier 等待)
        compute_time = 0
        result = {}
        if fabric.local_rank == 0:
            print(f"  [阶段2] 评估计算...")
            try:
                if eval_type == 'custom_verification4':
                    result, compute_time = timed_compute_custom_v4(evaluator, collection, collection_flip)
                elif eval_type == 'verification':
                    result, compute_time = timed_compute_verification(evaluator, collection, collection_flip)
                elif eval_type == 'tinyface':
                    result, compute_time = timed_compute_tinyface(evaluator, collection, collection_flip)
                else:
                    start = time.time()
                    result = evaluator.compute_metric(collection, collection_flip)
                    compute_time = time.time() - start

                print(f"  评估计算耗时: {compute_time:.2f}s")
            except Exception as e:
                print(f"  [错误] 评估计算失败: {e}")
                import traceback
                traceback.print_exc()
                timing_results.append({
                    'name': name, 'type': eval_type,
                    'extract_time': extract_time, 'compute_time': -1, 'total_time': -1,
                    'note': f'计算失败: {e}'
                })
                fabric.barrier()
                continue

        fabric.barrier()

        total_time = extract_time + compute_time
        timing_results.append({
            'name': name, 'type': eval_type,
            'extract_time': extract_time, 'compute_time': compute_time,
            'total_time': total_time, 'note': ''
        })

        del collection, collection_flip
        torch.cuda.empty_cache()

        # 超时检查: 15分钟
        if time.time() - total_start > 15 * 60:
            if fabric.local_rank == 0:
                print(f"\n[超时] 已运行 {(time.time()-total_start)/60:.1f} 分钟，停止")
            break

    # 汇总 (仅 rank 0 打印)
    total_elapsed = time.time() - total_start

    if fabric.local_rank == 0:
        print("\n")
        print("=" * 80)
        print("                    V1 评估流程分阶段计时汇总 (7 GPU DDP)")
        print("=" * 80)
        print(f"{'评估器':<20} {'类型':<22} {'特征提取(s)':<14} {'评估计算(s)':<14} {'总耗时(s)':<12} {'备注'}")
        print("-" * 80)

        total_extract = 0
        total_compute = 0
        for r in timing_results:
            ext_str = f"{r['extract_time']:.2f}" if r['extract_time'] >= 0 else "N/A"
            cmp_str = f"{r['compute_time']:.2f}" if r['compute_time'] >= 0 else "N/A"
            tot_str = f"{r['total_time']:.2f}" if r['total_time'] >= 0 else "N/A"
            print(f"{r['name']:<20} {r['type']:<22} {ext_str:<14} {cmp_str:<14} {tot_str:<12} {r['note']}")
            if r['extract_time'] >= 0:
                total_extract += r['extract_time']
            if r['compute_time'] >= 0:
                total_compute += r['compute_time']

        print("-" * 80)
        print(f"{'合计':<20} {'':<22} {total_extract:<14.2f} {total_compute:<14.2f} {total_extract+total_compute:<12.2f}")
        print(f"\n总运行时间: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
        print(f"模型加载: {model_load_time:.2f}s")
        total_sum = total_extract + total_compute
        if total_sum > 0:
            print(f"特征提取占比: {total_extract/total_sum*100:.1f}%")
            print(f"评估计算占比: {total_compute/total_sum*100:.1f}%")
        else:
            print("所有评估器均失败，无法计算占比")
        print("=" * 80)

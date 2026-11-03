"""
eval_models_report.py - 多模型 / 全数据集评估, 按各自评估方法跑, 结果存盘

复用 eval_all_trt_single.py 里验证过的全部流程:
  加载模型 -> 构建 TRT FP16 engine -> 多进程提特征 -> 按 evaluation_type 计算指标

与 launcher/single 的区别:
- 直接在 MODEL_PATHS 配置"几个指定模型"(不遍历某目录的所有 epoch)
- 遍历指定 eval_config(默认 report.yaml)里的所有数据集, 各按自己的
  evaluation_type 评估 (verification / ijbbc / ijbc_custom / tinyface /
  custom_verification4)
- 每个模型存一份完整原始结果 CSV (不经 summary 白名单过滤), 方便自己汇总
- custom_verification4 的 FAR 列扩展到 1e-4 ~ 1e-12

用法:
  cd .../work_0605
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH \
  python eval_models_report.py --num_gpu 7 --eval_config_name report
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
import time
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")

import numpy as np
np.bool = np.bool_

import torch
import torch.multiprocessing as mp
import pandas as pd
import sklearn.preprocessing

from models import get_model
from general_utils.config_utils import load_config

# 复用 single 脚本里已验证的全部底层函数
from eval_all_trt_single import (
    BATCH_SIZE, SHM_DIR,
    build_trt_engine,
    worker_extract, worker_extract_hf, worker_extract_hf_tinyface,
    gather_and_deduplicate, gather_and_deduplicate_hf, gather_and_deduplicate_tinyface,
    compute_metric_verification, compute_metric_ijbbc,
    compute_metric_ijbc_custom, compute_metric_tinyface,
)
from evaluations.cluster_utils import get_sim_matrix_large_scale_v5
from evaluations.custom_verification_evaluator import compute_tpir_from_hist


# ==================== 配置区: 在这里设置要测试的模型 ====================
# 每个元素为 (结果文件用的模型名, 模型目录路径)
# 模型目录需包含 model.pt 和 model.yaml
_PRETRAINED = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition'
MODEL_PATHS = [
    # ('adaface', f'{_PRETRAINED}/adaface_ir101_webface12m'),
    # ('s3_0925', f'{_PRETRAINED}/s3_0925'),
    # ('s3_1226', f'{_PRETRAINED}/s3_1226_14'),
    # ('s3_0323', f'{_PRETRAINED}/s3_full_0323_8'),
    # ('s4_0618', f'{_PRETRAINED}/s4_0618'),
    ('s4_0618_v2', f'{_PRETRAINED}/s4_0618_v2'),
    # 继续往下加要测的模型, 例如:
    # ('lvface', '/path/to/lvface'),
]

# custom_verification4 的目标 FAR (扩展到 1e-12)
CV4_TARGET_FARS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
# =====================================================================


def compute_metric_type4_ext(embeddings, query_ids, num_gpus):
    """custom_verification4 的扩展版: FAR 覆盖 1e-4 ~ 1e-12

    注意: get_sim_matrix_large_scale_v5 的 hist_bins 必须与
    compute_tpir_from_hist 的 hist_bins 一致, 否则累积/索引错位,
    TPIR 会被算成 ~100 (这是之前全 100.0 的根因)。
    沿用 single 脚本验证过的口径: hist_bins=2000 + precision='fp16'。
    """
    HIST_BINS = 2000
    pos_hist, neg_hist = get_sim_matrix_large_scale_v5(
        query_feats_list=embeddings,
        query_ids=query_ids,
        num_gpus=num_gpus,
        block_size=2048 * 16,
        show_progress=True,
        hist_bins=HIST_BINS,
        precision='fp16',
    )
    result, _ = compute_tpir_from_hist(
        pos_hist, neg_hist, hist_bins=HIST_BINS, target_fars=CV4_TARGET_FARS)
    print('  result:', result)
    return result


def eval_one_dataset(eval_name, info, eval_config, engine_path, num_gpu):
    """对单个数据集按其 evaluation_type 评估, 返回 result dict"""
    eval_data_path = os.path.join(eval_config.data_root, info.path)
    eval_type = info.evaluation_type
    metadata_path = os.path.join(eval_data_path, 'metadata.pt')

    shm_path = os.path.join(SHM_DIR, eval_name)
    if os.path.exists(shm_path):
        import shutil
        shutil.rmtree(shm_path, ignore_errors=True)
    os.makedirs(shm_path, exist_ok=True)

    # 选择 worker (与 single 一致)
    if eval_type == 'tinyface':
        worker_fn = worker_extract_hf_tinyface
    elif eval_type in ('ijbbc', 'ijbc_custom', 'verification'):
        worker_fn = worker_extract_hf
    else:
        worker_fn = worker_extract

    # 多进程提特征
    t0 = time.time()
    processes = []
    for rank in range(num_gpu):
        p = mp.Process(target=worker_fn,
                       args=(rank, num_gpu, engine_path, eval_data_path, shm_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"GPU {failed} 提取失败")
    print(f"  特征提取: {time.time()-t0:.1f}s ({num_gpu} GPU)")

    # 聚合 + 按类型计算指标 (与 single 一致, 仅 custom_verification4 换成扩展版)
    t0 = time.time()
    if eval_type == 'verification':
        features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, num_gpu)
        print(f"  样本数: {len(index)}")
        embeddings = (features_normal + features_flip).numpy()
        result = compute_metric_verification(embeddings, eval_data_path)

    elif eval_type == 'ijbbc':
        features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, num_gpu)
        print(f"  样本数: {len(index)}")
        embeddings = (features_normal + features_flip).numpy()
        result = compute_metric_ijbbc(embeddings, metadata_path)

    elif eval_type == 'ijbc_custom':
        features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, num_gpu)
        print(f"  样本数: {len(index)}")
        embeddings = (features_normal + features_flip).numpy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        real_indices = index.numpy()
        result = compute_metric_ijbc_custom(embeddings, real_indices, metadata_path, num_gpus=num_gpu)

    elif eval_type == 'tinyface':
        features_normal, features_flip, index, image_paths = gather_and_deduplicate_tinyface(shm_path, num_gpu)
        print(f"  样本数: {len(index)}")
        embeddings = (features_normal + features_flip).numpy()
        result = compute_metric_tinyface(embeddings, image_paths, metadata_path)

    else:
        # custom_verification4 / custom_verification 等
        features_normal, features_flip, labels = gather_and_deduplicate(shm_path, num_gpu)
        print(f"  样本数: {len(labels)}")
        embeddings = (features_normal + features_flip).numpy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        query_ids = labels.numpy()
        if eval_type == 'custom_verification4':
            result = compute_metric_type4_ext(embeddings, query_ids, num_gpus=num_gpu)
        else:
            from evaluations.custom_verification_evaluator import (
                generate_pairs_adaptive, find_tpir_at_far)
            from evaluations.verifications.verification import calculate_roc2
            dist, issame = generate_pairs_adaptive(embeddings, query_ids)
            thresholds = np.arange(0, 4, 0.01)
            tpr, fpr, accuracy = calculate_roc2(thresholds, dist, issame, nrof_folds=1)
            accuracy = accuracy * 100
            result = {'acc': float(np.mean(accuracy)), 'std': float(np.std(accuracy))}
            for far, tpir in zip([1e-6, 1e-5, 1e-4, 1e-3],
                                 find_tpir_at_far(tpr, fpr, target_fars=[1e-6, 1e-5, 1e-4, 1e-3])):
                result[f'tpir_at_far_{far}'] = tpir

    print(f"  指标计算: {time.time()-t0:.1f}s")
    print(f"  结果: {result}")

    # 清理本数据集的中间张量
    import gc
    for v in ('features_normal', 'features_flip', 'embeddings'):
        if v in locals():
            del locals()[v]
    gc.collect()
    torch.cuda.empty_cache()
    return result


def eval_one_model(model_dir, eval_config, num_gpu, precision='fp16'):
    """评估单个模型在 eval_config 全部数据集上的结果, 返回 {eval_name/metric: value}"""
    # 1) 加载模型 + 构建 TRT engine
    torch.cuda.set_device(0)
    model_config = load_config(os.path.join(model_dir, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(model_dir, 'model.pt'))
    model.eval()

    trt_cache = '/tmp/trt_report_cache'
    if os.path.exists(trt_cache):
        import shutil
        shutil.rmtree(trt_cache, ignore_errors=True)
    t0 = time.time()
    engine_path = build_trt_engine(model, batch_size=BATCH_SIZE, cache_dir=trt_cache,
                                   precision=precision)
    if engine_path is None:
        raise RuntimeError("TRT 构建失败")
    print(f"  TRT engine 构建: {time.time()-t0:.1f}s (precision={precision})")
    del model
    torch.cuda.empty_cache()

    # 2) 遍历全部数据集
    all_result = {}
    for eval_name, info in eval_config.per_epoch_evaluations.items():
        print(f"\n{'-'*50}")
        print(f"  数据集: {eval_name} (type={info.evaluation_type})")
        print(f"{'-'*50}")
        try:
            result = eval_one_dataset(eval_name, info, eval_config, engine_path, num_gpu)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  数据集 {eval_name} 失败: {e}")
            continue
        all_result.update({f'{eval_name}/{k}': v for k, v in result.items()})

    # 清理
    import shutil
    shutil.rmtree(trt_cache, ignore_errors=True)
    shutil.rmtree(SHM_DIR, ignore_errors=True)
    torch.cuda.empty_cache()
    return all_result


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--eval_config_name', type=str, default='report')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp16', 'fp32'],
                        help="TRT engine 精度: fp16(默认, 快) / fp32(更稳, 更接近 PyTorch)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果目录, 默认 eval3_results/report_<config>_<precision>')
    args = parser.parse_args()

    eval_config = load_config(f'evaluations/configs/{args.eval_config_name}.yaml')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(
        script_dir, 'eval3_results', f'report_{args.eval_config_name}_{args.precision}')
    os.makedirs(output_dir, exist_ok=True)

    # 用于跨模型合并的总表 (列=模型, 行=指标)
    combined = {}

    for name, model_dir in MODEL_PATHS:
        print(f"\n{'='*60}")
        print(f"评估模型: {name}  ({model_dir})")
        print(f"{'='*60}")
        if not os.path.exists(os.path.join(model_dir, 'model.pt')):
            print(f"  跳过: 未找到 {model_dir}/model.pt")
            continue
        try:
            all_result = eval_one_model(model_dir, eval_config, args.num_gpu,
                                        precision=args.precision)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  模型 {name} 失败: {e}")
            continue

        # 保存该模型的完整原始结果 (所有 key, 不过滤)
        per_model_csv = os.path.join(output_dir, f'{name}.csv')
        pd.DataFrame(pd.Series(all_result), columns=['val']).to_csv(
            per_model_csv, encoding='utf-8-sig')
        print(f"\n  已保存: {per_model_csv}")
        combined[name] = all_result

    if not combined:
        print("\n没有成功评估的模型, 退出。")
        sys.exit(1)

    # 跨模型合并总表: 行=指标 (eval_name/metric), 列=模型名
    combined_df = pd.DataFrame(combined)
    combined_df.index.name = 'metric'
    combined_csv = os.path.join(output_dir, 'combined_all_metrics.csv')
    combined_df.to_csv(combined_csv, encoding='utf-8-sig')
    print(f"\n{'='*60}")
    print(f"全部指标合并表已保存: {combined_csv}")
    print(f"(行=指标, 列=模型; 各数据集结果均在此, 自行挑列汇总)")
    print(f"{'='*60}")

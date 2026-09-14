"""两遍直方图法验证: 解决 1e-10/1e-9 端点的 bin 分辨率问题

遍1: 200k bins 全范围 [-1,1]  → 常规 far + 定位右尾
遍2: 200 万 bins, 范围缩到右尾 (分辨率 ~5e-10) → 精确 far=1e-10/1e-9/1e-8
"""
import os
import sys
import time

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import sklearn.preprocessing
import torch
import torch.multiprocessing as mp

from evaluations.cluster_utils import get_sim_matrix_large_scale_v6
from evaluations.custom_verification_evaluator import compute_tpir_from_hist
from ijbc_v6hist_poc import extract_feats, build_query_ids, eval_001_subset, TARGET_FARS

FINE_BINS = 200_000
MIN_FAR = 1e-10


def two_pass_v6(embeddings, query_ids, num_gpu):
    # 两遍全 fp32: 1e-10/1e-9 端点负分在 0.9995+, tf32 误差(~1e-3) 会淹没右尾;
    # 堆版 v3 引擎同为 fp32 GEMM, 数值分布一致
    t0 = time.time()
    pos_h, neg_h = get_sim_matrix_large_scale_v6(
        query_feats_list=embeddings, query_ids=query_ids, num_gpus=num_gpu,
        block_size=2048 * 16, hist_bins=200_000, hist_range=(-1.0, 1.0),
        precision='fp32', skip_clamp=True, show_progress=False)
    res, thr = compute_tpir_from_hist(pos_h, neg_h, hist_bins=200_000,
                                      hist_range=(-1.0, 1.0), target_fars=TARGET_FARS)
    t1 = time.time()
    total_pos_global = int(pos_h.sum())
    total_neg_global = int(neg_h.sum())

    # 遍2: 右尾精细直方图, 窗口按 1e-7 阈值定位 (覆盖 far<=1e-7 的全部右尾)
    # skip_clamp 关闭 → v6 内部 clamp_(lo,hi), 越界对堆积在下边界 bin (idx=0),
    # searchsorted 只要不落入堆积 bin 即为真值
    lo = float(thr[1e-7]) - 5e-4
    hi = 1.0002
    pos_f, neg_f = get_sim_matrix_large_scale_v6(
        query_feats_list=embeddings, query_ids=query_ids, num_gpus=num_gpu,
        block_size=2048 * 16, hist_bins=FINE_BINS, hist_range=(lo, hi),
        precision='fp32', skip_clamp=False, show_progress=False)

    neg_cum = np.cumsum(neg_f[::-1])
    pos_cum = np.cumsum(pos_f[::-1])
    edges = np.linspace(lo, hi, FINE_BINS + 1)
    res_f = {}
    for far in [1e-10, 1e-9, 1e-8, 1e-7]:
        target = far * total_neg_global
        idx = int(np.searchsorted(neg_cum, target, side='left'))
        if idx >= FINE_BINS - 1:  # 落入(或触及)堆积 bin → 窗口内真值不足, 回退遍1
            res_f[f'tpir_at_far_{far}'] = res.get(f'tpir_at_far_{far}')
            print(f"  [warn] far={far}: 窗口内真值负对不足 (idx={idx}), 回退遍1值")
        else:
            res_f[f'tpir_at_far_{far}'] = float(pos_cum[idx]) / total_pos_global * 100
    t2 = time.time()
    print(f"  遍1(fp32 200k): {t1-t0:.1f}s, 遍2(fp32 {FINE_BINS}k, [{lo:.6f},{hi}]): {t2-t1:.1f}s")
    return res, res_f, t2 - t0


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7
    import polars as pl
    old = {r['far']: r for r in pl.read_parquet(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'ijbc_v6hist_poc.parquet')).to_dicts()}

    print("提取 IJBC 特征 (7 GPU)...")
    f_n, f_f, index = extract_feats(num_gpu)
    embeddings = (f_n + f_f).numpy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    real_indices = index.numpy()
    del f_n, f_f
    meta = torch.load(os.path.join(DATA := '/data1/dataset_0605/facerec_val/IJBC_gt_aligned',
                                   'metadata.pt'), weights_only=False)
    query_ids = build_query_ids(index, meta)

    print("\n[all 集] 两遍直方图 vs 旧堆引擎:")
    res, res_f, t_total = two_pass_v6(embeddings, query_ids, num_gpu)
    for far in TARGET_FARS:
        heap = old[far]['heap'] if far in old else float('nan')
        v6_1p = old[far]['v6hist'] if far in old else float('nan')
        v6_2p = res_f.get(f'tpir_at_far_{far}', res[f'tpir_at_far_{far}'])
        print(f"  {far:>8}: 堆 {heap:8.4f} | 单遍 {v6_1p:8.4f} | 两遍 {v6_2p:8.4f} "
              f"| 两遍-堆 {v6_2p-heap:+.4f}")
    print(f"  总耗时 {t_total:.1f}s (旧堆引擎 89.8s)")

    sub = eval_001_subset(embeddings, query_ids, real_indices, num_gpu, 1e-3)
    if sub is not None:
        emb1, ids1 = sub
        print(f"\n[001 子集 {len(ids1):,} 张] 两遍直方图 vs 旧堆引擎:")
        res1, res_f1, t1 = two_pass_v6(emb1, ids1, num_gpu)
        for far in TARGET_FARS:
            v6_2p = res_f1.get(f'tpir_at_far_{far}', res1[f'tpir_at_far_{far}'])
            # 旧堆引擎 001 值从 fork 轮 raw csv 读
            print(f"  {far:>8}: 两遍 {v6_2p:8.4f}")
        print(f"  总耗时 {t1:.1f}s")


if __name__ == '__main__':
    main()

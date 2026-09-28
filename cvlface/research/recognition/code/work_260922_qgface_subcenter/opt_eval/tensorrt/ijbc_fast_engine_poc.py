"""新快速引擎 vs 旧堆引擎 对拍 PoC (ijbc_custom)

新: get_sim_matrix_topk_hist_fast (fp32 单遍, 直方图+精确 topk 混合 TPIR)
旧: get_sim_matrix_batch_balanced_silent + compute_tpir_from_heap
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sklearn.preprocessing
import torch
import torch.multiprocessing as mp
import polars as pl

from evaluations.cluster_utils import (get_sim_matrix_batch_balanced_silent,
                                       get_sim_matrix_topk_hist_fast,
                                       compute_tpir_topk_hist)
from evaluations.custom_ijbbc_evaluator import compute_tpir_from_heap
from ijbc_v6hist_poc import extract_feats, build_query_ids, eval_001_subset, TARGET_FARS

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7

    print("提取 IJBC 特征 (7 GPU)...")
    f_n, f_f, index = extract_feats(num_gpu)
    embeddings = (f_n + f_f).numpy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    real_indices = index.numpy()
    del f_n, f_f
    meta = torch.load(os.path.join(DATA, 'metadata.pt'), weights_only=False)
    query_ids = build_query_ids(index, meta)
    N = len(query_ids)
    _, counts = np.unique(query_ids, return_counts=True)
    total_pos = int((counts * (counts - 1) // 2).sum())
    total_neg = N * (N - 1) // 2 - total_pos
    topk_old = max(int(total_neg * 1e-3), 1000)
    print(f"N={N:,} 正对 {total_pos:,} 负对 {total_neg:,}")

    rows = []
    for tag, emb, ids in [('all', embeddings, query_ids)]:
        t0 = time.time()
        p_old, n_old, _ = get_sim_matrix_batch_balanced_silent(
            query_feats_list=emb, query_ids=ids, num_gpus=num_gpu,
            block_size=2048 * 5, topk=topk_old, threshold=None,
            show_progress=False, return_stats_only=False, return_pairs_only=False)
        t_old = time.time() - t0
        res_old, thr_old = compute_tpir_from_heap(n_old, p_old, total_neg, TARGET_FARS)
        del p_old, n_old

        t0 = time.time()
        ph, nh, tk = get_sim_matrix_topk_hist_fast(emb, ids, num_gpus=num_gpu)
        res_new, thr_new = compute_tpir_topk_hist(
            ph, nh, tk, total_neg, TARGET_FARS)
        t_new = time.time() - t0
        print(f"\n[{tag}] 旧 {t_old:.1f}s vs 新 {t_new:.1f}s ({t_old/t_new:.1f}x)")
        print(f"pos_hist 总数 {int(ph.sum()):,} (真值 {total_pos:,})")
        print(f"{'far':>8} {'旧':>10} {'新':>10} {'差':>9}")
        for far in TARGET_FARS:
            a = res_old[f'tpir_at_far_{far}']
            b = res_new[f'tpir_at_far_{far}']
            print(f"{far:>8} {a:10.4f} {b:10.4f} {b-a:+9.4f}")
            rows.append({'set': tag, 'far': far, 'heap': a, 'fast': b,
                         'sec_old': round(t_old, 1), 'sec_new': round(t_new, 1)})
        del ph, nh, tk

    sub = eval_001_subset(embeddings, query_ids, real_indices, num_gpu, 1e-3)
    if sub is not None:
        emb1, ids1 = sub
        n1 = len(ids1)
        tp1 = int((lambda c: (c * (c - 1) // 2).sum())(np.unique(ids1, return_counts=True)[1]))
        tn1 = n1 * (n1 - 1) // 2 - tp1
        k1 = max(int(tn1 * 1e-3), 1000)
        t0 = time.time()
        p1, n1s, _ = get_sim_matrix_batch_balanced_silent(
            query_feats_list=emb1, query_ids=ids1, num_gpus=num_gpu,
            block_size=2048 * 5, topk=k1, threshold=None,
            show_progress=False, return_stats_only=False, return_pairs_only=False)
        t_old = time.time() - t0
        res_old, _ = compute_tpir_from_heap(n1s, p1, tn1, TARGET_FARS)
        del p1, n1s
        t0 = time.time()
        ph, nh, tk = get_sim_matrix_topk_hist_fast(emb1, ids1, num_gpus=num_gpu)
        res_new, _ = compute_tpir_topk_hist(ph, nh, tk, tn1, TARGET_FARS)
        t_new = time.time() - t0
        print(f"\n[001 {n1:,}] 旧 {t_old:.1f}s vs 新 {t_new:.1f}s ({t_old/t_new:.1f}x)")
        for far in TARGET_FARS:
            a = res_old[f'tpir_at_far_{far}']
            b = res_new[f'tpir_at_far_{far}']
            print(f"{far:>8} {a:10.4f} {b:10.4f} {b-a:+9.4f}")
            rows.append({'set': '001', 'far': far, 'heap': a, 'fast': b,
                         'sec_old': round(t_old, 1), 'sec_new': round(t_new, 1)})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'ijbc_fast_engine_poc.parquet')
    pl.DataFrame(rows).write_parquet(out)
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()

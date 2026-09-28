"""ijbc_custom topk 路径 v6 直方图化 PoC: 旧堆引擎 vs v6 直方图引擎 对拍

旧路径: get_sim_matrix_batch_balanced_silent (v3 引擎: fp32 GEMM 无 tf32,
        每 tile 3 次布尔 mask 副本, 正对 fp16 逐块传 CPU, topk*2 粗筛有损)
新路径: get_sim_matrix_large_scale_v6 (tf32 + skip_clamp + 200k bins 直方图,
        1e-5 分辨率) + compute_tpir_from_hist

验证: 两条路径 TPIR 逐项对拍 + 耗时
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

from evaluations.cluster_utils import (get_sim_matrix_batch_balanced_silent,
                                       get_sim_matrix_large_scale_v6)
from evaluations.custom_verification_evaluator import compute_tpir_from_hist
from evaluations.custom_ijbbc_evaluator import get_pairs_data, compute_tpir_from_heap

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
TARGET_FARS = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_feats(num_gpu):
    """复用生产 worker 提取 IJBC 特征 (fork 轮实测 ~33s)"""
    import eval_all_trt_single as E
    shm = '/dev/shm/ijbc_poc'
    os.makedirs(shm, exist_ok=True)
    procs = []
    for rank in range(num_gpu):
        p = mp.Process(target=E.worker_extract_hf,
                       args=(rank, num_gpu, ENGINE, DATA, shm))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    f_n, f_f, index = E.gather_and_deduplicate_hf(shm, num_gpu)
    import shutil
    shutil.rmtree(shm, ignore_errors=True)
    return f_n, f_f, index


def build_query_ids(index, meta):
    for k in meta:
        if torch.is_tensor(meta[k]):
            meta[k] = meta[k].numpy()
    group_map = get_pairs_data(meta)
    templates = meta['templates']
    index_docid_list = [group_map[templates[i]] for i in range(len(templates))]
    return np.array([index_docid_list[idx] for idx in index])


def eval_001_subset(embeddings, query_ids, real_indices, num_gpus, max_far):
    """001 子集 (与生产逻辑一致)"""
    list_path = os.path.join(SCRIPT_DIR, '001_ijbc_image_list.txt')
    if not os.path.exists(list_path):
        return None
    with open(list_path) as f:
        names = [l.strip() for l in f.readlines()]
    real_idx_to_row = {}
    for row, idx in enumerate(real_indices):
        if idx not in real_idx_to_row:
            real_idx_to_row[idx] = row
    sel = []
    for name in names:
        try:
            fi = int(os.path.splitext(name)[0]) - 1
            if fi in real_idx_to_row:
                sel.append(real_idx_to_row[fi])
        except (ValueError, KeyError):
            pass
    sel = np.array(sel)
    if len(sel) == 0:
        return None
    return embeddings[sel], query_ids[sel]


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
    total_pairs = N * (N - 1) // 2
    _, counts = np.unique(query_ids, return_counts=True)
    total_pos = int((counts * (counts - 1) // 2).sum())
    total_neg = total_pairs - total_pos
    max_far = max(TARGET_FARS)
    topk = max(int(total_neg * max_far), 1000)
    print(f"N={N:,}, 正对 {total_pos:,}, 负对 {total_neg:,}, topk={topk:,}")

    # ---- 旧路径: v3 堆引擎 ----
    print("\n[旧] get_sim_matrix_batch_balanced_silent (v3 堆引擎)...")
    t0 = time.time()
    pos_scores, neg_scores, _ = get_sim_matrix_batch_balanced_silent(
        query_feats_list=embeddings, query_ids=query_ids, num_gpus=num_gpu,
        block_size=2048 * 5, topk=topk, threshold=None, show_progress=False,
        return_stats_only=False, return_pairs_only=False)
    t_old = time.time() - t0
    res_old, _ = compute_tpir_from_heap(neg_scores, pos_scores, total_neg, TARGET_FARS)
    print(f"[旧] {t_old:.1f}s, 正对 {len(pos_scores):,}, 堆 {len(neg_scores):,}")

    # ---- 新路径: v6 直方图 ----
    print("\n[新] get_sim_matrix_large_scale_v6 + 直方图 TPIR (200k bins)...")
    t0 = time.time()
    pos_hist, neg_hist = get_sim_matrix_large_scale_v6(
        query_feats_list=embeddings, query_ids=query_ids, num_gpus=num_gpu,
        block_size=2048 * 16, hist_bins=200_000, hist_range=(-1.0, 1.0),
        precision='tf32', skip_clamp=True, show_progress=False)
    res_new, thr_new = compute_tpir_from_hist(
        pos_hist, neg_hist, hist_bins=200_000, hist_range=(-1.0, 1.0),
        target_fars=TARGET_FARS)
    t_new = time.time() - t0
    pos_hist_sum = int(pos_hist.sum())
    print(f"[新] {t_new:.1f}s, pos_hist 总数 {pos_hist_sum:,} (真值 {total_pos:,}, "
          f"差 {pos_hist_sum - total_pos:+,})")

    # ---- 对拍 (all) ----
    print(f"\n{'far':>8} {'旧(堆)':>10} {'新(v6直方图)':>12} {'差':>8}")
    for far in TARGET_FARS:
        a, b = res_old[f'tpir_at_far_{far}'], res_new[f'tpir_at_far_{far}']
        print(f"{far:>8} {a:10.4f} {b:12.4f} {b-a:+8.4f}")
    print(f"\n耗时: 旧 {t_old:.1f}s vs 新 {t_new:.1f}s ({t_old/t_new:.2f}x)")

    # ---- 001 子集对拍 ----
    sub = eval_001_subset(embeddings, query_ids, real_indices, num_gpu, max_far)
    if sub is not None:
        emb1, ids1 = sub
        n1 = len(ids1)
        tp1 = int((lambda c: (c * (c - 1) // 2).sum())(np.unique(ids1, return_counts=True)[1]))
        tn1 = n1 * (n1 - 1) // 2 - tp1
        k1 = max(int(tn1 * max_far), 1000)
        t0 = time.time()
        p1o, n1o, _ = get_sim_matrix_batch_balanced_silent(
            query_feats_list=emb1, query_ids=ids1, num_gpus=num_gpu,
            block_size=2048 * 5, topk=k1, threshold=None, show_progress=False,
            return_stats_only=False, return_pairs_only=False)
        t_old1 = time.time() - t0
        res_old1, _ = compute_tpir_from_heap(n1o, p1o, tn1, TARGET_FARS)
        t0 = time.time()
        ph1, nh1 = get_sim_matrix_large_scale_v6(
            query_feats_list=emb1, query_ids=ids1, num_gpus=num_gpu,
            block_size=2048 * 16, hist_bins=200_000, hist_range=(-1.0, 1.0),
            precision='tf32', skip_clamp=True, show_progress=False)
        res_new1, _ = compute_tpir_from_hist(ph1, nh1, hist_bins=200_000,
                                             hist_range=(-1.0, 1.0),
                                             target_fars=TARGET_FARS)
        t_new1 = time.time() - t0
        print(f"\n[001 子集 {n1:,} 张] 旧 {t_old1:.1f}s vs 新 {t_new1:.1f}s")
        for far in TARGET_FARS:
            a, b = res_old1[f'tpir_at_far_{far}'], res_new1[f'tpir_at_far_{far}']
            print(f"{far:>8} {a:10.4f} {b:12.4f} {b-a:+8.4f}")

    import polars as pl
    rows = []
    for far in TARGET_FARS:
        rows.append({'far': far, 'heap': res_old[f'tpir_at_far_{far}'],
                     'v6hist': res_new[f'tpir_at_far_{far}'],
                     'sec_old': round(t_old, 1), 'sec_new': round(t_new, 1)})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'ijbc_v6hist_poc.parquet')
    pl.DataFrame(rows).write_parquet(out)
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()

"""ijbc 降 bins 验证: v7@2000 (新统一配置) vs v6@200k (原配置) 的 TPIR 对拍"""
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

import eval_all_trt_single as E
from evaluations.cluster_utils import (get_sim_matrix_large_scale_v7,
                                       get_sim_matrix_large_scale_v6)
from evaluations.custom_verification_evaluator import compute_tpir_from_hist

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
TARGET_FARS = [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 1e-5, 1e-4, 1e-3]


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7

    shm = '/dev/shm/ijbc_bins_verify'
    os.makedirs(shm, exist_ok=True)
    procs = []
    for rank in range(num_gpu):
        p = mp.Process(target=E.worker_extract_hf,
                       args=(rank, num_gpu, '/tmp/int8_ptq_work/engine_fp16.plan',
                             DATA, shm))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    f_n, f_f, index = E.gather_and_deduplicate_hf(shm, num_gpu)
    import shutil
    shutil.rmtree(shm, ignore_errors=True)
    embeddings = sklearn.preprocessing.normalize((f_n + f_f).numpy())
    real_indices = index.numpy()
    print(f"提特征: {len(index):,} 张")

    # query_ids (与 compute_metric_ijbc_custom 一致)
    from evaluations.custom_ijbbc_evaluator import get_pairs_data
    meta = torch.load(os.path.join(DATA, 'metadata.pt'), weights_only=False)
    for k in meta:
        if torch.is_tensor(meta[k]):
            meta[k] = meta[k].numpy()
    group_map = get_pairs_data(meta)
    templates = meta['templates']
    docids = [group_map[templates[i]] for i in range(len(templates))]
    query_ids = np.array([docids[idx] for idx in real_indices])
    _, counts = np.unique(query_ids, return_counts=True)
    total_pos = int((counts * (counts - 1) // 2).sum())
    print(f"正对 {total_pos:,}")

    t0 = time.time()
    p6, n6 = get_sim_matrix_large_scale_v6(
        query_feats_list=embeddings, query_ids=query_ids, num_gpus=num_gpu,
        block_size=2048 * 16, show_progress=False, hist_bins=200_000,
        hist_range=(-1.0, 1.0), precision='tf32', skip_clamp=True)
    t_v6 = time.time() - t0
    r6, _ = compute_tpir_from_hist(p6, n6, hist_bins=200_000,
                                   hist_range=(-1.0, 1.0), target_fars=TARGET_FARS)

    t0 = time.time()
    p7, n7 = get_sim_matrix_large_scale_v7(
        query_feats_list=embeddings, query_ids=query_ids, num_gpus=num_gpu,
        block_size=16384, hist_bins=2000, hist_range=(-1.0, 1.0))
    t_v7 = time.time() - t0
    r7, _ = compute_tpir_from_hist(p7, n7, hist_bins=2000,
                                   hist_range=(-1.0, 1.0), target_fars=TARGET_FARS)

    print(f"\nv6@200k {t_v6:.1f}s vs v7@2000 {t_v7:.1f}s ({t_v6/t_v7:.1f}x)")
    print(f"posΣ: v6={int(p6.sum()):,} v7={int(p7.sum()):,} 真值={total_pos:,}")
    print(f"{'far':>8} {'v6@200k':>10} {'v7@2000':>10} {'差':>9}")
    mx = 0.0
    for far in TARGET_FARS:
        a, b = r6[f'tpir_at_far_{far}'], r7[f'tpir_at_far_{far}']
        d = b - a
        mx = max(mx, abs(d)) if far >= 1e-8 else mx
        print(f"{far:>8} {a:10.4f} {b:10.4f} {d:+9.4f}")
    print(f"\nfar>=1e-8 端点最大偏差: {mx:.4f}")


if __name__ == '__main__':
    main()

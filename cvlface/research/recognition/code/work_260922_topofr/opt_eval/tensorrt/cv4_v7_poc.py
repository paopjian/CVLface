"""cv4 enhance 单集: v6 (tf32+histc) vs v7 (fp16 GEMM+CUDA 双桶核) TPIR 对拍"""
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

DATA = '/data1/dataset_0605/test_1201'
TARGET_FARS = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


def extract(num_gpu):
    shm = '/dev/shm/cv4_poc'
    os.makedirs(shm, exist_ok=True)
    procs = []
    for rank in range(num_gpu):
        p = mp.Process(target=E.worker_extract,
                       args=(rank, num_gpu, '/tmp/int8_ptq_work/engine_fp16.plan',
                             DATA, shm))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    f_n, f_f, labels = E.gather_and_deduplicate(shm, num_gpu)
    import shutil
    shutil.rmtree(shm, ignore_errors=True)
    emb = (f_n + f_f).numpy()
    emb = sklearn.preprocessing.normalize(emb)
    return emb, labels.numpy()


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7
    emb, ids = extract(num_gpu)
    print(f"enhance: {len(ids):,} 张")

    t0 = time.time()
    p6, n6 = get_sim_matrix_large_scale_v6(
        query_feats_list=emb, query_ids=ids, num_gpus=num_gpu,
        block_size=2048 * 16, show_progress=False, hist_bins=2000,
        precision='tf32', skip_clamp=True)
    t_v6 = time.time() - t0
    r6, _ = compute_tpir_from_hist(p6, n6, hist_bins=2000,
                                   hist_range=(-1.0, 1.0), target_fars=TARGET_FARS)

    t0 = time.time()
    p7, n7 = get_sim_matrix_large_scale_v7(emb, ids, num_gpus=num_gpu,
                                      hist_bins=2000)
    t_v7 = time.time() - t0
    r7, _ = compute_tpir_from_hist(p7, n7, hist_bins=2000,
                                   hist_range=(-1.0, 1.0), target_fars=TARGET_FARS)

    same_p = np.array_equal(p6, p7)
    print(f"\nv6 {t_v6:.1f}s vs v7 {t_v7:.1f}s ({t_v6/t_v7:.2f}x)")
    print(f"hist 逐 bin 一致: pos={same_p} neg={np.array_equal(n6, n7)}")
    print(f"posΣ v6={int(p6.sum()):,} v7={int(p7.sum()):,}")
    print(f"{'far':>8} {'v6':>10} {'v7':>10} {'差':>9}")
    for far in TARGET_FARS:
        a, b = r6[f'tpir_at_far_{far}'], r7[f'tpir_at_far_{far}']
        print(f"{far:>8} {a:10.4f} {b:10.4f} {b-a:+9.4f}")


if __name__ == '__main__':
    main()

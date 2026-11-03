"""cv4 enhance: v7 C 模式 (fused_he 融合提取) vs v6 提取 对拍
对拍维度: hist 一致性 / neg 对数量与坐标集合重合率 / 分数差 / 耗时"""
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
from evaluations.cluster_utils import (get_sim_matrix_large_scale_v6,
                                       get_sim_matrix_large_scale_v7)

DATA = '/data1/dataset_0605/test_enhance'
THR = 0.5


def extract(num_gpu):
    shm = '/dev/shm/cv4_c_poc'
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
    return sklearn.preprocessing.normalize(emb), labels.numpy()


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7
    emb, ids = extract(num_gpu)
    print(f"enhance: {len(ids):,} 张, neg_threshold={THR}")

    t0 = time.time()
    p6, n6, pairs6 = get_sim_matrix_large_scale_v6(
        query_feats_list=emb, query_ids=ids, num_gpus=num_gpu,
        block_size=2048 * 8, show_progress=False,
        collect_pairs_config={'sample_type': 'neg', 'threshold_mode': 'above',
                              'threshold': THR, 'max_pairs': -1})
    t_v6 = time.time() - t0

    t0 = time.time()
    p7, n7, neg7, pos7 = get_sim_matrix_large_scale_v7(
        query_feats_list=emb, query_ids=ids, num_gpus=num_gpu,
        block_size=16384, hist_bins=2000, neg_threshold=THR)
    t_v7 = time.time() - t0

    print(f"\nv6 提取 {t_v6:.1f}s vs v7 C 模式 {t_v7:.1f}s ({t_v6/t_v7:.2f}x)")
    print(f"hist 逐 bin: pos={np.array_equal(p6, p7)} neg={np.array_equal(n6, n7)}")
    print(f"posΣ v6={int(p6.sum()):,} v7={int(p7.sum()):,}")
    print(f"neg 对数: v6={len(pairs6):,} v7={len(neg7):,}")

    # 坐标集合重合率 (i,j 键); 分数差
    k6 = set((int(a) << 32 | int(b)) for a, b, _ in pairs6)
    k7 = set((int(a) << 32 | int(b)) for a, b, _ in neg7)
    inter = len(k6 & k7)
    union = len(k6 | k7)
    print(f"坐标集合: 交 {inter:,} / 并 {union:,} = {inter/union*100:.2f}% 重合"
          f" (fp16 vs tf32 GEMM 边界对进出)")
    # 共同键的分数差
    d6 = {(int(a) << 32 | int(b)): float(s) for a, b, s in pairs6}
    d7 = {(int(a) << 32 | int(b)): float(s) for a, b, s in neg7}
    diffs = [abs(d6[k] - d7[k]) for k in (k6 & k7)]
    if diffs:
        print(f"共同对分数差: mean={np.mean(diffs):.2e} max={np.max(diffs):.2e}")
    print(f"v7 pos 提取 (pos<={0.2}): {len(pos7):,} 对 (同 ID 低分/漏检)")


if __name__ == '__main__':
    main()

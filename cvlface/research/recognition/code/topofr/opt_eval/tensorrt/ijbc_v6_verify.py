"""端到端验证: 提特征 + 新 compute_metric_ijbc_custom(v6), 对比 fork 轮历史值"""
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
import pandas as pd
import sklearn.preprocessing
import torch
import torch.multiprocessing as mp

import eval_all_trt_single as E

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7

    t0 = time.time()
    shm = '/dev/shm/ijbc_verify'
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
    print(f"提特征: {time.time()-t0:.1f}s, {len(index):,} 张")

    embeddings = (f_n + f_f).numpy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    t0 = time.time()
    res = E.compute_metric_ijbc_custom(embeddings, index.numpy(),
                                       os.path.join(DATA, 'metadata.pt'), num_gpu)
    t_metric = time.time() - t0
    print(f"指标计算(v6): {t_metric:.1f}s")

    old = pd.read_csv('eval3_results/fastdecode_test_0913_fork/epoch_0_raw.csv',
                      index_col=0)['val']
    print(f"\n{'key':>34} {'fork轮(旧引擎)':>14} {'v6版':>10} {'差':>9}")
    bad = 0
    for k, v in res.items():
        ck = f'ijbc/{k}'
        o = old.get(ck)
        d = (v - o) if o is not None else float('nan')
        flag = '' if (o is None or abs(d) < 0.5) else '  <-- 大偏差'
        if flag:
            bad += 1
        print(f"{k:>34} {o if o is not None else float('nan'):14.4f} {v:10.4f} {d:+9.4f}{flag}")
    print(f"\n大偏差(>0.5)项数: {bad} (预期: 仅 1e-10/1e-9/5e-7 端点)")


if __name__ == '__main__':
    main()

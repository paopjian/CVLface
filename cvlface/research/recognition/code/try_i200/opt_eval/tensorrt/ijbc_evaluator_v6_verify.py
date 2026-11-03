"""验证: CustomIJBCEvaluator.compute_metric (v6, 训练期评估用) 与
eval_all_trt_single v6 版结果对拍 (两者应几乎一致, 同为 v6 tf32 同特征)"""
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
from evaluations.custom_ijbbc_evaluator import CustomIJBCEvaluator

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    num_gpu = 7

    # 提特征 (复用生产 worker)
    shm = '/dev/shm/ijbc_eval_verify'
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
    print(f"提特征: {len(index):,} 张")

    # 构造 evaluator (跳过 __init__: 不建 fabric/数据集, 只注入 meta)
    ev = CustomIJBCEvaluator.__new__(CustomIJBCEvaluator)
    ev.name = 'ijbc'
    ev.meta = torch.load(os.path.join(DATA, 'metadata.pt'), weights_only=False)

    t0 = time.time()
    res = ev.compute_metric({'features': f_n, 'index': index},
                            {'features': f_f, 'index': index})
    t_metric = time.time() - t0
    print(f"compute_metric (v6): {t_metric:.1f}s")

    # 对拍参考: fork 轮 eval_all_trt_single v6 版的 ijbc 段
    old = pd.read_csv('eval3_results/fastdecode_test_0913_fork/epoch_0_raw.csv',
                      index_col=0)['val']
    bad = 0
    for k, v in res.items():
        o = old.get(f'ijbc/{k}')
        d = (v - o) if o is not None else float('nan')
        flag = '  <-- 超0.5' if (o is not None and abs(d) > 0.5) else ''
        if flag:
            bad += 1
        print(f"{k:>34} ref={o:9.4f} new={v:9.4f} diff={d:+.4f}{flag}")
    print(f"\n超0.5偏差项: {bad}")


if __name__ == '__main__':
    main()

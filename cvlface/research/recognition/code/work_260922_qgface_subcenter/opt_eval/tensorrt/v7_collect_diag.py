"""诊断: v7 C 模式提取对的坐标/分数真伪 (抽样现算验证)"""
import os
import sys

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
from evaluations.cluster_utils import get_sim_matrix_large_scale_v7

DATA = '/data1/dataset_0605/test_enhance'


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    shm = '/dev/shm/diag_v7'
    os.makedirs(shm, exist_ok=True)
    procs = []
    for rank in range(7):
        p = mp.Process(target=E.worker_extract,
                       args=(rank, 7, '/tmp/int8_ptq_work/engine_fp16.plan', DATA, shm))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    f_n, f_f, labels = E.gather_and_deduplicate(shm, 7)
    import shutil
    shutil.rmtree(shm, ignore_errors=True)
    emb = sklearn.preprocessing.normalize((f_n + f_f).numpy())
    ids = labels.numpy()
    codes = np.unique(ids, return_inverse=True)[1].astype(np.int32)

    neg_t, pos_t, neg7, pos7 = get_sim_matrix_large_scale_v7(
        emb, ids, num_gpus=7, hist_bins=2000, neg_threshold=0.5)
    print(f"v7 neg 对数: {len(neg7):,}")
    s = neg7[:, 2]
    print(f"分数分布: >=0.5 占 {np.mean(s >= 0.5)*100:.1f}%, "
          f"min={s.min():.3f}, max={s.max():.3f}")

    ft = torch.from_numpy(emb)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(neg7), 20, replace=False)
    bad_score = bad_same = 0
    for k in pick:
        i, j, sv = int(neg7[k, 0]), int(neg7[k, 1]), float(neg7[k, 2])
        real = float(ft[i] @ ft[j])
        same = codes[i] == codes[j]
        if abs(real - sv) > 0.01:
            bad_score += 1
        if same:
            bad_same += 1
        if k < 5 or abs(real - sv) > 0.01:
            print(f"  ({i},{j}) 记录 {sv:.4f} 现算 {real:.4f} 同ID={same}")
    print(f"抽样 20: 分数不符 {bad_score}, 同ID {bad_same}")


if __name__ == '__main__':
    main()

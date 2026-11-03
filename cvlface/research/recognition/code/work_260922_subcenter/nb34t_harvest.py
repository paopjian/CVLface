"""
nb34t_harvest.py — s4_0922_sub 在 test_34t 的负对阈值邻域打捞

对 1e-6..1e-12 每档: 取阈值上方(FA侧)最贴线 100 对 + 阈值下方最贴线 100 对。
实现: v7 捞对模式单遍收集全部 sim >= (1e-6阈值-0.004) 的跨ID对, 再分层切片。

产出 (out_dir):
  nb_1e-6.csv ... nb_1e-12.csv   每档 200 对 (side=above/below, 按贴线程度排序)
  fa_counts.csv                  各档 FA 总数 (hist 口径, 供核对)
  fa_pairs_all.parquet           打捞全量明细 (≥ 0.364)
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
os.environ.setdefault('TRT_BATCH_SIZE', '256')
os.environ.setdefault('TRT_NUM_WORKERS', '5')
os.environ.setdefault('TRT_DECODE', 'auto')
os.environ.setdefault('TRT_ENGINE_BS', '512')

import time
import shutil
import argparse

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import sklearn.preprocessing

from models import get_model
from general_utils.config_utils import load_config
from evaluations.cluster_utils import get_sim_matrix_large_scale_v7
from eval_all_trt_single import (
    worker_extract,
    gather_and_deduplicate,
    build_trt_engine,
    SHM_DIR,
)

HIST_BINS = 2000
HIST_LO, HIST_HI = -1.0, 1.0
# s4_0922_sub 在 test_34t 的实测阈值 (eval3_results/s4_0922_sub_34t_v2, 2026-10-27)
LEVELS = {1e-6: 0.368, 1e-7: 0.412, 1e-8: 0.452, 1e-9: 0.487,
          1e-10: 0.514, 1e-11: 0.538, 1e-12: 0.561}
NEG_COLLECT_THR = 0.364   # = thr(1e-6) - 0.004, 保证 1e-6 档 below 侧有足够候选


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/data1/dataset_260918/test_34t')
    parser.add_argument('--ckpt_path',
                        default='/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/s4_0922_sub')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--per_side', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shm_path = os.path.join(SHM_DIR, 'nb34t')
    os.makedirs(shm_path, exist_ok=True)

    # ---- engine + 特征 ----
    torch.cuda.set_device(0)
    model_config = load_config(os.path.join(args.ckpt_path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(args.ckpt_path, 'model.pt'))
    model.eval()
    t0 = time.time()
    engine_path = build_trt_engine(model, batch_size=512,
                                   cache_dir='/tmp/trt_eval3_cache',
                                   precision='fp16')
    del model
    torch.cuda.empty_cache()
    print(f"[1] engine: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    procs = []
    for rank in range(args.num_gpu):
        p = mp.Process(target=worker_extract,
                       args=(rank, args.num_gpu, engine_path,
                             args.dataset, shm_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    assert not failed, f'GPU {failed} 提取失败'
    print(f"[2] 特征提取: {time.time()-t0:.1f}s", flush=True)

    fn, ff, labels_t = gather_and_deduplicate(shm_path, args.num_gpu)
    labels = labels_t.numpy()
    N = len(labels)
    print(f"    样本数: {N:,}", flush=True)
    embeddings = sklearn.preprocessing.normalize((fn + ff).numpy())

    tsv = pd.read_csv(os.path.join(args.dataset, 'train.tsv'),
                      sep='\t', header=None)
    tsv.columns = ['_idx', 'path', 'label']
    paths = np.asarray(tsv['path'].tolist())
    assert len(paths) == N

    # ---- v7 单遍: hist + 捞对 ----
    t0 = time.time()
    pos_hist, neg_hist, neg_pairs, pos_pairs = get_sim_matrix_large_scale_v7(
        query_feats_list=embeddings, query_ids=labels,
        num_gpus=args.num_gpu, block_size=16384,
        hist_bins=HIST_BINS, hist_range=(HIST_LO, HIST_HI),
        neg_threshold=NEG_COLLECT_THR)
    print(f"[3] v7 捞对(≥{NEG_COLLECT_THR}): {time.time()-t0:.1f}s, "
          f"共 {len(neg_pairs):,} 对", flush=True)
    total_neg = int(neg_hist.sum())

    dfp = pd.DataFrame(neg_pairs, columns=['i', 'j', 'score'])
    dfp = dfp.sort_values('score', ascending=False).reset_index(drop=True)
    dfp['i'] = dfp['i'].astype(np.int64)
    dfp['j'] = dfp['j'].astype(np.int64)
    dfp['path_i'] = paths[dfp['i'].values]
    dfp['path_j'] = paths[dfp['j'].values]
    dfp['label_i'] = labels[dfp['i'].values]
    dfp['label_j'] = labels[dfp['j'].values]
    dfp.to_parquet(os.path.join(args.out_dir, 'fa_pairs_all.parquet'),
                   index=False)

    # ---- 各档计数 + 邻域 ±100 ----
    rows = []
    score_v = dfp['score'].values
    for far in sorted(LEVELS, reverse=True):
        thr = LEVELS[far]
        b = int(np.floor((thr - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS))
        cnt_hist = int(neg_hist[b:].sum())
        cnt_pairs = int((score_v >= thr).sum())
        rows.append({'far': far, 'threshold': thr,
                     'fa_count_hist': cnt_hist, 'fa_count_pairs': cnt_pairs,
                     'expected': far * total_neg})
        print(f"    {far:g} thr {thr}: hist {cnt_hist:,} / 实捞 {cnt_pairs:,} "
              f"(期望 {far*total_neg:,.0f})", flush=True)

        above = dfp[score_v >= thr].nsmallest(args.per_side, 'score') \
            if cnt_pairs else dfp.iloc[0:0]
        below = dfp[score_v < thr].nlargest(args.per_side, 'score')
        above = above.copy()
        below = below.copy()
        above['side'] = 'above'
        below['side'] = 'below'
        nb = pd.concat([above, below], ignore_index=True)
        nb.insert(0, 'far', far)
        nb.insert(1, 'threshold', thr)
        nb.to_csv(os.path.join(args.out_dir, f'nb_{far:g}.csv'), index=False)
        print(f"    nb_{far:g}.csv: above {len(above)} / below {len(below)}",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'fa_counts.csv'),
                              index=False)

    shutil.rmtree(shm_path, ignore_errors=True)
    if os.path.exists('/tmp/trt_eval3_cache'):
        shutil.rmtree('/tmp/trt_eval3_cache')
    print("test_34t 阈值邻域打捞完成!", flush=True)

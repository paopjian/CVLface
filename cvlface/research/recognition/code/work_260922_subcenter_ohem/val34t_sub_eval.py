"""
val34t_sub_eval.py — 基准模型 s4_0922_sub 在 val_34t 上的本地评估 + FA/极值打捞

产出 (out_dir):
  tpir_fpir.json          七档 (1e-6..1e-12) TPIR@FPIR + 阈值 (与 001 包同口径: 真值分母)
  feats_val34t.npz        fp16 特征 + labels (供复算)
  neg_top100_high.csv     负对相似度最高 top100 (跨ID)
  neg_top100_low.csv      负对相似度最低 top100 (跨ID)
  pos_top100_high.csv     正对相似度最高 top100 (同ID)
  pos_top100_low.csv      正对相似度最低 top100 (同ID)
  fa_*/cmp_*.csv          新001 相同图片对 + subcenter 分数 (+旧001 有则并入)
  summary.json
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

import sys
import time
import json
import shutil
import threading
import argparse
import queue as queue_mod

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import sklearn.preprocessing

from models import get_model
from general_utils.config_utils import load_config
from evaluations.cluster_utils import get_sim_matrix_large_scale_v7
from evaluations.custom_verification_evaluator import compute_tpir_from_hist
from eval_all_trt_single import (
    worker_extract,
    gather_and_deduplicate,
    build_trt_engine,
    SHM_DIR,
)

HIST_BINS = 2000
HIST_LO, HIST_HI = -1.0, 1.0
LEVELS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]  # 深→浅倒序同旧包风格


def extremes_pass(feats, codes, N, num_gpus, block_size=16384, K=100):
    """自定义极值 pass: 全 N×N 分块, 取 pos 最高 K 对 / neg 最低 K 对 (fp16 同 v7 口径)"""
    nb = (N + block_size - 1) // block_size
    tiles = [(bi, bj) for bi in range(nb) for bj in range(bi, nb)]
    qs = [queue_mod.Queue() for _ in range(num_gpus)]
    for t in tiles:
        qs[t[1] % num_gpus].put(t)
    for q in qs:
        q.put(None)
    out = {k: [[] for _ in range(num_gpus)] for k in ('pos_high', 'neg_low')}

    def _worker(g, q):
        with torch.cuda.device(g):
            dev = torch.device(f'cuda:{g}')
            shard = {}
            for bj in range(g, nb, num_gpus):
                c0, c1 = bj * block_size, min((bj + 1) * block_size, N)
                shard[bj] = (feats[c0:c1].to(dev), codes[c0:c1].to(dev))
            torch.cuda.current_stream().synchronize()
            cache = (-1, None, None)
            NEG_INF, POS_INF = -2.0, 2.0
            while True:
                it = q.get()
                if it is None:
                    break
                bi, bj = it
                r0, r1 = bi * block_size, min((bi + 1) * block_size, N)
                c0, c1 = bj * block_size, min((bj + 1) * block_size, N)
                diag = bi == bj
                if diag:
                    b1, d1 = shard[bi]
                    b2, d2 = b1, d1
                else:
                    if cache[0] == bi:
                        b1, d1 = cache[1], cache[2]
                    else:
                        b1 = feats[r0:r1].to(dev)
                        d1 = codes[r0:r1].to(dev)
                        cache = (bi, b1, d1)
                    b2, d2 = shard[bj]
                sim = torch.matmul(b1, b2.T)
                C = sim.shape[1]
                same = d1[:, None] == d2[None, :]
                tri = torch.ones_like(same).triu(diagonal=1) if diag else None

                m = torch.where(same, sim, torch.full_like(sim, NEG_INF))
                if diag:
                    m = torch.where(tri, m, torch.full_like(m, NEG_INF))
                v, idx = torch.topk(m.flatten(), min(4 * K, m.numel()))
                sel = v > NEG_INF / 2
                v, idx = v[sel].float(), idx[sel]
                if v.numel():
                    ii = idx // C + r0
                    jj = idx % C + c0
                    out['pos_high'][g].append(
                        torch.stack([v, ii.float(), jj.float()], dim=1).cpu())
                del m

                m2 = torch.where(same, torch.full_like(sim, POS_INF), sim)
                if diag:
                    m2 = torch.where(tri, m2, torch.full_like(m2, POS_INF))
                v2, idx2 = torch.topk(m2.flatten(), min(4 * K, m2.numel()),
                                      largest=False)
                sel2 = v2 < POS_INF / 2
                v2, idx2 = v2[sel2].float(), idx2[sel2]
                if v2.numel():
                    ii = idx2 // C + r0
                    jj = idx2 % C + c0
                    out['neg_low'][g].append(
                        torch.stack([v2, ii.float(), jj.float()], dim=1).cpu())
                del m2, sim
            del shard
            torch.cuda.empty_cache()

    ths = [threading.Thread(target=_worker, args=(g, qs[g]))
           for g in range(num_gpus)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    res = {}
    for key in ('pos_high', 'neg_low'):
        parts = [a for g in range(num_gpus) for a in out[key] if a is not None]
        if parts:
            arr = torch.cat(parts, dim=0).numpy()
            order = np.argsort(-arr[:, 0]) if key == 'pos_high' \
                else np.argsort(arr[:, 0])
            arr = arr[order][:K]
            res[key] = arr
        else:
            res[key] = np.zeros((0, 3), dtype=np.float32)
    return res


def pair_rows(arr, labels, paths):
    df = pd.DataFrame(arr, columns=['score', 'i', 'j'])
    df['i'] = df['i'].astype(np.int64)
    df['j'] = df['j'].astype(np.int64)
    df['score'] = df['score'].astype(np.float64)
    df['label_i'] = labels[df['i'].values]
    df['label_j'] = labels[df['j'].values]
    df['path_i'] = paths[df['i'].values]
    df['path_j'] = paths[df['j'].values]
    return df[['score', 'i', 'j', 'label_i', 'label_j', 'path_i', 'path_j']]


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/data1/dataset_260918/val_34t')
    parser.add_argument('--ckpt_path',
                        default='/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/s4_0922_sub')
    parser.add_argument('--refs_dir', required=True,
                        help='新001 清单 CSV + old001_check.json 所在目录')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--num_gpu', type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shm_path = os.path.join(SHM_DIR, 'val34t_sub')
    os.makedirs(shm_path, exist_ok=True)

    # ---- 1) engine + 特征 ----
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
    feats = torch.from_numpy(embeddings).half().contiguous()
    np.savez_compressed(os.path.join(args.out_dir, 'feats_val34t.npz'),
                        feats=feats.numpy(), labels=labels)

    tsv = pd.read_csv(os.path.join(args.dataset, 'train.tsv'),
                      sep='\t', header=None)
    tsv.columns = ['_idx', 'path', 'label']
    paths = np.asarray(tsv['path'].tolist())
    assert len(paths) == N

    # ---- 2) v7 hist → 阈值/TPIR ----
    t0 = time.time()
    pos_hist, neg_hist = get_sim_matrix_large_scale_v7(
        query_feats_list=embeddings, query_ids=labels,
        num_gpus=args.num_gpu, block_size=16384,
        hist_bins=HIST_BINS, hist_range=(HIST_LO, HIST_HI))
    result, thresholds = compute_tpir_from_hist(
        pos_hist, neg_hist, hist_bins=HIST_BINS,
        hist_range=(HIST_LO, HIST_HI), target_fars=LEVELS)
    print(f"[3] v7 hist: {time.time()-t0:.1f}s", flush=True)
    print(f"    result: {result}", flush=True)
    print(f"    thresholds: {thresholds}", flush=True)
    total_neg = int(neg_hist.sum())
    total_pos = int(pos_hist.sum())
    with open(os.path.join(args.out_dir, 'tpir_fpir.json'), 'w') as f:
        json.dump({'tpir': result,
                   'thresholds': {f'{k:g}': v for k, v in thresholds.items()},
                   'total_neg_pairs': total_neg,
                   'total_pos_pairs': total_pos,
                   'n_samples': int(N)}, f, ensure_ascii=False, indent=2)

    thr_deep_to_shallow = {lv: thresholds[lv] for lv in LEVELS}
    thr_1e6 = thr_deep_to_shallow[1e-6]

    # ---- 3) v7 collect: neg 高尾 (≥ 1e-6阈值-0.004) + pos 低尾 (左累积~2000) ----
    pos_cum = np.cumsum(pos_hist)
    s_low = float(np.searchsorted(pos_cum, 2000) / HIST_BINS * 2.0 - 1.0)
    neg_collect_thr = round(thr_1e6 - 0.004, 3)
    t0 = time.time()
    pos_hist2, neg_hist2, neg_pairs, pos_pairs = get_sim_matrix_large_scale_v7(
        query_feats_list=embeddings, query_ids=labels,
        num_gpus=args.num_gpu, block_size=16384,
        hist_bins=HIST_BINS, hist_range=(HIST_LO, HIST_HI),
        neg_threshold=neg_collect_thr, pos_threshold=s_low)
    print(f"[4] v7 collect (neg≥{neg_collect_thr}, pos≤{s_low:.3f}): "
          f"{time.time()-t0:.1f}s, neg {len(neg_pairs):,} / pos {len(pos_pairs):,}",
          flush=True)

    if len(neg_pairs):
        order = np.argsort(-neg_pairs[:, 2])
        df = pair_rows(neg_pairs[order[:100]], labels, paths)
        df.insert(0, 'rank', np.arange(1, len(df) + 1))
        df.to_csv(os.path.join(args.out_dir, 'neg_top100_high.csv'), index=False)
    if len(pos_pairs):
        order = np.argsort(pos_pairs[:, 2])
        df = pair_rows(pos_pairs[order[:100]], labels, paths)
        df.insert(0, 'rank', np.arange(1, len(df) + 1))
        df.to_csv(os.path.join(args.out_dir, 'pos_top100_low.csv'), index=False)

    # ---- 4) 自定义极值 pass: pos 最高 / neg 最低 ----
    t0 = time.time()
    codes = np.unique(labels, return_inverse=True)[1].astype(np.int32)
    codes_t = torch.from_numpy(codes)
    ex = extremes_pass(feats, codes_t, N, args.num_gpu)
    print(f"[5] 极值 pass: {time.time()-t0:.1f}s "
          f"(pos_high {len(ex['pos_high'])}, neg_low {len(ex['neg_low'])})",
          flush=True)
    if len(ex['pos_high']):
        df = pair_rows(ex['pos_high'], labels, paths)
        df.insert(0, 'rank', np.arange(1, len(df) + 1))
        df.to_csv(os.path.join(args.out_dir, 'pos_top100_high.csv'), index=False)
    if len(ex['neg_low']):
        df = pair_rows(ex['neg_low'], labels, paths)
        df.insert(0, 'rank', np.arange(1, len(df) + 1))
        df.to_csv(os.path.join(args.out_dir, 'neg_top100_low.csv'), index=False)

    # ---- 5) 新001 相同图片对打分 ----
    emb = embeddings.astype(np.float32)
    old001 = {}
    oj = os.path.join(args.refs_dir, 'old001_check.json')
    if os.path.isfile(oj):
        with open(oj) as f:
            for p in json.load(f).get('pairs', []):
                if p.get('old001_score') is not None:
                    old001[(int(p['rec_idx_a']), int(p['rec_idx_b']))] = \
                        p['old001_score']
    print(f"    旧001 可比对数: {len(old001)}", flush=True)

    def score_pairs(a_idx, b_idx):
        return np.einsum('ij,ij->i', emb[a_idx], emb[b_idx])

    jobs = {
        'fa_pairs_1e-09.csv': 'fa_list_1e-09.csv',
        'neg_top100_001_cmp.csv': 'neg_top100.csv',
        'pos_lowest_top100_001_cmp.csv': 'pos_lowest_top100.csv',
        'pos_near_miss1000_001_cmp.csv': 'pos_near_miss_top1000.csv',
    }
    for out_name, ref_name in jobs.items():
        rp = os.path.join(args.refs_dir, ref_name)
        if not os.path.isfile(rp):
            print(f"    缺参考清单 {ref_name}, 跳过", flush=True)
            continue
        df = pd.read_csv(rp)
        a = df['rec_idx_a'].values.astype(np.int64)
        b = df['rec_idx_b'].values.astype(np.int64)
        df['sub_score'] = score_pairs(a, b)
        df['old001_score'] = [old001.get((x, y), np.nan)
                              for x, y in zip(a, b)]
        df.to_csv(os.path.join(args.out_dir, out_name), index=False)
        print(f"    {out_name}: {len(df)} 对 (旧001 匹配 "
              f"{df['old001_score'].notna().sum()})", flush=True)

    shutil.rmtree(shm_path, ignore_errors=True)
    if os.path.exists('/tmp/trt_eval3_cache'):
        shutil.rmtree('/tmp/trt_eval3_cache')
    print("val34t 本地评估完成!", flush=True)

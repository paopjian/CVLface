"""
val34t_fix_extremes.py — 从 feats_val34t.npz 续跑: 极值 pass(pos最高/neg最低 top100)
+ 新001 相同图片对打分, 输出直接写 /root/zhaokj/work_eval/260922_val_34t
(主脚本 extremes_pass 的 parts 合并少了 [g] 一层索引导致 TypeError, 其余产物已落盘)
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
import json
import shutil
import threading
import queue as queue_mod
import argparse

import numpy as np
import pandas as pd
import torch

SRC = '/data2/zkj-data/sub_fa/val34t'
REFS = '/data2/zkj-data/sub_fa/001_refs'


def extremes_pass(feats, codes, N, num_gpus, block_size=16384, K=100):
    """全 N×N 分块: pos 最高 K 对 / neg 最低 K 对 (fp16, 与 v7 同口径)"""
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
        parts = [t for g in range(num_gpus) for t in out[key][g]]  # 逐张量展平
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='/root/zhaokj/work_eval/260922_val_34t')
    parser.add_argument('--dataset', default='/data1/dataset_260918/val_34t')
    parser.add_argument('--num_gpu', type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(os.path.join(SRC, 'feats_val34t.npz'))
    feats = torch.from_numpy(d['feats'])          # fp16 (N,512) 已归一化
    labels = d['labels'].astype(np.int64)
    N = len(labels)
    print(f'载入特征: {N:,} × {feats.shape[1]}', flush=True)

    tsv = pd.read_csv(os.path.join(args.dataset, 'train.tsv'),
                      sep='\t', header=None)
    tsv.columns = ['_idx', 'path', 'label']
    paths = np.asarray(tsv['path'].tolist())
    assert len(paths) == N

    # ---- 极值 pass ----
    codes = np.unique(labels, return_inverse=True)[1].astype(np.int32)
    codes_t = torch.from_numpy(codes)
    ex = extremes_pass(feats, codes_t, N, args.num_gpu)
    print(f"极值: pos_high {len(ex['pos_high'])}, neg_low {len(ex['neg_low'])}",
          flush=True)
    for key, name in (('pos_high', 'pos_top100_high.csv'),
                      ('neg_low', 'neg_top100_low.csv')):
        if len(ex[key]):
            df = pair_rows(ex[key], labels, paths)
            df.insert(0, 'rank', np.arange(1, len(df) + 1))
            df.to_csv(os.path.join(args.out_dir, name), index=False)

    # ---- 新001 相同图片对打分 ----
    emb = feats.float().numpy()
    old001 = {}
    with open(os.path.join(REFS, 'old001_check.json')) as f:
        for p in json.load(f).get('pairs', []):
            if p.get('old001_score') is not None:
                old001[(int(p['rec_idx_a']), int(p['rec_idx_b']))] = \
                    p['old001_score']
    print(f'旧001 可比对数: {len(old001)}', flush=True)

    jobs = {
        'fa_pairs_1e-09.csv': 'fa_list_1e-09.csv',
        'neg_top100_001_cmp.csv': 'neg_top100.csv',
        'pos_lowest_top100_001_cmp.csv': 'pos_lowest_top100.csv',
        'pos_near_miss1000_001_cmp.csv': 'pos_near_miss_top1000.csv',
    }
    for out_name, ref_name in jobs.items():
        rp = os.path.join(REFS, ref_name)
        if not os.path.isfile(rp):
            print(f'缺 {ref_name}, 跳过', flush=True)
            continue
        df = pd.read_csv(rp)
        a = df['rec_idx_a'].values.astype(np.int64)
        b = df['rec_idx_b'].values.astype(np.int64)
        df['sub_score'] = np.einsum('ij,ij->i', emb[a], emb[b])
        df['old001_score'] = [old001.get((x, y), np.nan) for x, y in zip(a, b)]
        df.to_csv(os.path.join(args.out_dir, out_name), index=False)
        print(f"{out_name}: {len(df)} 对 (旧001 匹配 {df['old001_score'].notna().sum()})",
              flush=True)

    # ---- 把主脚本已产出的一并搬到 work_eval ----
    for f in ('tpir_fpir.json', 'neg_top100_high.csv', 'pos_top100_low.csv',
              'run.log', 'feats_val34t.npz'):
        s = os.path.join(SRC, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out_dir, f))
    print('完成! 产物在', args.out_dir, flush=True)

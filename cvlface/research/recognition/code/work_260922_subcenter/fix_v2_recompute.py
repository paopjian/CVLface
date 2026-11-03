#!/root/miniconda3/envs/cvlface/bin/python
"""fix_v2: 重算 v7 捞对产出的两个 CSV (原列错位+astype截断毁掉了真分数),
并双向匹配旧001 分数, 然后重绘图"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
import json
import numpy as np
import pandas as pd
import torch

from evaluations.cluster_utils import get_sim_matrix_large_scale_v7

D = '/root/zhaokj/work_eval/260922_val_34t'
SRC = '/data2/zkj-data/sub_fa/val34t'
REFS = '/data2/zkj-data/sub_fa/001_refs'
DS_PATH = '/data1/dataset_260918/val_34t'

d = np.load(os.path.join(SRC, 'feats_val34t.npz'))
feats = d['feats']                      # fp16 (N,512), 已是归一化后的 embedding
labels = d['labels'].astype(np.int64)
N = len(labels)
emb = feats.astype(np.float32)
tsv = pd.read_csv(os.path.join(DS_PATH, 'train.tsv'), sep='\t', header=None)
tsv.columns = ['_idx', 'path', 'label']
paths = np.asarray(tsv['path'].tolist())
assert len(paths) == N
print(f'载入: N={N:,}', flush=True)


def write_pairs(pairs, name, asc):
    # pairs: (M,3) 列 = (i, j, score)  ← v7 布局
    df = pd.DataFrame({'i': pairs[:, 0].astype(np.int64),
                       'j': pairs[:, 1].astype(np.int64),
                       'score': pairs[:, 2].astype(np.float64)})
    df = df.sort_values('score', ascending=asc).reset_index(drop=True)
    df = df.head(100).copy()
    df.insert(0, 'rank', np.arange(1, len(df) + 1))
    df['label_i'] = labels[df['i'].values]
    df['label_j'] = labels[df['j'].values]
    df['path_i'] = paths[df['i'].values]
    df['path_j'] = paths[df['j'].values]
    df.to_csv(os.path.join(D, name), index=False)
    print(f'{name}: [{df["score"].min():.4f}, {df["score"].max():.4f}], '
          f'top1 i/j={df.iloc[0]["i"]}/{df.iloc[0]["j"]}', flush=True)
    return df


# ---- call A: neg 高尾 ----
pos_hist, neg_hist, neg_pairs, _ = get_sim_matrix_large_scale_v7(
    query_feats_list=emb, query_ids=labels, num_gpus=7, block_size=16384,
    hist_bins=2000, hist_range=(-1.0, 1.0), neg_threshold=0.364)
print(f'neg≥0.364: {len(neg_pairs):,}', flush=True)
neg_hi = write_pairs(neg_pairs, 'neg_top100_high.csv', asc=False)

# ---- call B: pos 低尾 (阈值从 hist 左累积 2000 处取) ----
pos_cum = np.cumsum(pos_hist)
b_low = int(np.searchsorted(pos_cum, 2000))
s_low = b_low / 2000.0 * 2.0 - 1.0
_, _, _, pos_pairs = get_sim_matrix_large_scale_v7(
    query_feats_list=emb, query_ids=labels, num_gpus=7, block_size=16384,
    hist_bins=2000, hist_range=(-1.0, 1.0), pos_threshold=s_low)
print(f'pos≤{s_low:.3f}: {len(pos_pairs):,}', flush=True)
pos_lo = write_pairs(pos_pairs, 'pos_top100_low.csv', asc=True)

# ---- 双向匹配旧001 ----
old001 = {}
with open(os.path.join(REFS, 'old001_check.json')) as f:
    for p in json.load(f).get('pairs', []):
        if p.get('old001_score') is not None:
            a, b = int(p['rec_idx_a']), int(p['rec_idx_b'])
            old001[(a, b)] = p['old001_score']
            old001[(b, a)] = p['old001_score']

for out_name in ('fa_pairs_1e-09.csv', 'neg_top100_001_cmp.csv',
                 'pos_lowest_top100_001_cmp.csv', 'pos_near_miss1000_001_cmp.csv'):
    p = os.path.join(D, out_name)
    if not os.path.isfile(p):
        continue
    df = pd.read_csv(p)
    df['old001_score'] = [old001.get((x, y), np.nan)
                          for x, y in zip(df['rec_idx_a'], df['rec_idx_b'])]
    df.to_csv(p, index=False)
    print(f'{out_name}: 旧001 双向匹配 {int(df["old001_score"].notna().sum())}',
          flush=True)
print('fix_v2 完成', flush=True)

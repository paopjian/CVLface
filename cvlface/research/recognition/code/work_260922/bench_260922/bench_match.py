"""NxN 匹配路径对比 (同一份特征缓存, 8 卡):
  1. cluster_utils v7 (现行 custom_verification4 用的 CUDA fp16 双桶核)
  2. portable_eval pos_neg_hist_fused (v7 同款核的 portable 移植)
  3. portable_eval pos_neg_hist_torch (纯 torch 优化版)

用法: python bench_match.py --feats ../bench_260922/results --tag test_glint_nv8 --gpus 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORK, 'portable_eval'))
sys.path.insert(0, WORK)

import eval_lib as el  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True, help='特征缓存目录')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gpus', type=int, default=8)
    ap.add_argument('--fars', default='1e-5,1e-6,1e-7,1e-8,1e-9,1e-10')
    ap.add_argument('--only', default='', help='逗号分隔: v7,fused,torch (默认全跑)')
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(',') if x.strip()]
    fars = [float(x) for x in args.fars.split(',')]

    feats, ids, meta = el.load_feature_cache(args.feats, args.tag)
    N = feats.shape[0]
    print(f'特征: {feats.shape} {feats.dtype}, ids {ids.shape}, '
          f"pipeline={meta.get('pipeline')}, 对数 {N * (N - 1) // 2:,}")

    results = {}

    def run(name, fn, tpir_fn):
        if only and name not in only:
            print(f'[{name}] 跳过 (--only)')
            return
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        r = tpir_fn(out)
        results[name] = {'time_s': round(dt, 1),
                         'G_pairs_per_s': round(N * (N - 1) / 2 / dt / 1e9, 1),
                         'tpir': r}
        k5 = [k for k in r if '1e-05' in k or '1e-5' in k]
        print(f'[{name}] {dt:.1f}s ({results[name]["G_pairs_per_s"]} G对/s) '
              f'TPIR@1e-5={r[k5[0]] if k5 else float("nan"):.3f}')

    from evaluations.cluster_utils import get_sim_matrix_large_scale_v7
    from evaluations.custom_verification_evaluator import compute_tpir_from_hist
    from functools import partial

    run('v7',
        partial(get_sim_matrix_large_scale_v7, query_feats_list=feats,
                query_ids=ids, num_gpus=args.gpus, block_size=16384,
                hist_bins=2000, hist_range=(-1.0, 1.0)),
        lambda out: compute_tpir_from_hist(out[0], out[1], hist_bins=2000,
                                           hist_range=(-1.0, 1.0),
                                           target_fars=fars)[0])

    run('fused',
        partial(el.pos_neg_hist_fused, feats, ids, num_gpus=args.gpus,
                block=16384, bins=2000),
        lambda out: el.tpir_from_hist(out[0], out[1], 2000, fpirs=fars)[0])

    run('torch',
        partial(el.pos_neg_hist_torch, feats, ids, num_gpus=args.gpus,
                block=8192, bins=2000),
        lambda out: el.tpir_from_hist(out[0], out[1], 2000, fpirs=fars)[0])

    out = os.path.join(args.feats, f'bench_match_{args.tag}.json')
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'已保存: {out}')


if __name__ == '__main__':
    main()

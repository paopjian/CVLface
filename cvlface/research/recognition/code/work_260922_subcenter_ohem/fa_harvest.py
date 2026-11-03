"""
fa_harvest.py - 按已评估阈值反向打捞 FA 对 (False Accept pairs)

复用 eval_all_trt_single 的特征提取链路 (保证特征与评估时同源同预处理),
再用 v7 引擎的捞对模式 (neg_threshold) 单遍全矩阵收集所有
跨 ID 且 sim >= neg_threshold 的负对 (i, j, score)。

产出 (out_dir):
  fa_counts.csv      各档 FPIR 的 FA 对数 (hist 口径 + 实捞口径交叉验证)
  fa_pairs.parquet   全部 FA 对明细 (按分数降序, 含双端路径/标签)
  sample_pairs/      按分数从高到低前 N 对拼图 (肉眼核查撞脸)
  summary.json       汇总 (总量/档位计数/震中图 Top)
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
import sys
import time
import json
import shutil

# 必须在 import eval_all_trt_single 之前设置:
# 模块导入时即读取这些环境变量 (BATCH_SIZE / ENGINE_BS 为模块级常量)
os.environ.setdefault('TRT_BATCH_SIZE', '256')
os.environ.setdefault('TRT_NUM_WORKERS', '5')
os.environ.setdefault('TRT_DECODE', 'auto')
os.environ.setdefault('TRT_ENGINE_BS', '512')

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
    FastImageFolderDataset,
    SHM_DIR,
)

HIST_BINS = 2000
HIST_LO, HIST_HI = -1.0, 1.0


def load_paths(kind, dataset_path):
    """数据集行号 → 原图路径 (34t: rec 内嵌, 取 tsv 登记的原始相对路径; 3t: 文件直读)"""
    if kind == 'rec':
        info = pd.read_csv(os.path.join(dataset_path, 'train.tsv'),
                           sep='\t', header=None)
        info.columns = ['_idx', 'path', 'label']
        return info['path'].tolist()
    ds = FastImageFolderDataset(dataset_path)
    return [p for p, _ in ds.samples]


def export_samples(kind, dataset_path, dfp, sdir, n_pairs):
    """按分数降序导出前 n 对拼图 (左右并排)"""
    import cv2
    os.makedirs(sdir, exist_ok=True)
    mx = None
    if kind == 'rec':
        from dataset.base_dataset import MXFaceDataset
        mx = MXFaceDataset(root_dir=dataset_path, local_rank=0)

    def _img(row_idx):
        if kind == 'rec':
            pil, _ = mx.read_sample(int(row_idx))
            return cv2.cvtColor(np.asarray(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        return cv2.imread(row_idx)

    exported = 0
    for _, r in dfp.head(n_pairs).iterrows():
        try:
            a, b = _img(r['path_i']), _img(r['path_j'])
            if a is None or b is None:
                continue
            h = max(a.shape[0], b.shape[0])
            def _pad(x):
                if x.shape[0] == h:
                    return x
                pad = np.zeros((h - x.shape[0], x.shape[1], 3), dtype=x.dtype)
                return np.vstack([x, pad])
            canvas = np.concatenate([_pad(a), _pad(b)], axis=1)
            name = (f"{exported:04d}_s{r['score']:.4f}_"
                    f"{r['label_i']}-{r['label_j']}.jpg")
            cv2.imwrite(os.path.join(sdir, name), canvas)
            exported += 1
        except Exception as e:
            print(f"  样例导出失败 {r['i']}-{r['j']}: {e}", flush=True)
    if mx is not None:
        mx.dispose()
    return exported


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--kind', choices=['rec', 'folder'], required=True)
    parser.add_argument('--neg_threshold', type=float, required=True,
                        help='打捞阈值: 收集全部 sim>=该值的跨ID对 (取需要完整名单的最浅档)')
    parser.add_argument('--levels', required=True,
                        help='档位:阈值 列表 "far:thr,...", 用于分层计数')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--sample_pairs', type=int, default=200,
                        help='按分数从高到低导出前 N 对拼图 (0=不导)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_name = os.path.basename(os.path.normpath(args.out_dir))

    # ---- 1) TRT engine (与评估同参 fp16) ----
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
    assert engine_path, 'TRT engine 构建失败'
    print(f"[1/4] engine 构建: {time.time()-t0:.1f}s", flush=True)

    # ---- 2) 特征提取 (与评估同链路, 含 nvjpeg 优先/cv2 降级) ----
    shm_path = os.path.join(SHM_DIR, eval_name)
    os.makedirs(shm_path, exist_ok=True)
    t0 = time.time()
    procs = []
    for rank in range(args.num_gpu):
        p = mp.Process(target=worker_extract,
                       args=(rank, args.num_gpu, engine_path, args.dataset, shm_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    assert not failed, f'GPU {failed} 特征提取失败'
    print(f"[2/4] 特征提取: {time.time()-t0:.1f}s", flush=True)

    fn, ff, labels = gather_and_deduplicate(shm_path, args.num_gpu)
    n = len(labels)
    print(f"      样本数: {n:,}", flush=True)
    embeddings = sklearn.preprocessing.normalize((fn + ff).numpy())
    query_ids = labels.numpy()

    # ---- 3) v7 全矩阵 + 捞对 (fp16 GEMM, 与阈值计算同源) ----
    t0 = time.time()
    pos_hist, neg_hist, neg_pairs, pos_pairs = get_sim_matrix_large_scale_v7(
        query_feats_list=embeddings,
        query_ids=query_ids,
        num_gpus=args.num_gpu,
        block_size=16384,
        hist_bins=HIST_BINS,
        hist_range=(HIST_LO, HIST_HI),
        neg_threshold=args.neg_threshold,
    )
    print(f"[3/4] sim+捞对: {time.time()-t0:.1f}s, "
          f"FA(>= {args.neg_threshold}) = {len(neg_pairs):,}", flush=True)

    total_neg = int(neg_hist.sum())
    total_pos = int(pos_hist.sum())
    print(f"      neg 对总数 {total_neg:,} / pos 对总数 {total_pos:,}", flush=True)

    # ---- 4) 分层计数 + 明细落盘 ----
    t0 = time.time()
    levels = []
    for kv in args.levels.split(','):
        far_s, thr_s = kv.split(':')
        levels.append((float(far_s), float(thr_s)))
    levels.sort(key=lambda x: x[0], reverse=True)  # 深→浅

    rows = []
    for far, thr in levels:
        b = int(np.floor((thr - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS))
        cnt_hist = int(neg_hist[b:].sum())
        cnt_pairs = int((neg_pairs[:, 2] >= thr).sum()) if len(neg_pairs) else 0
        rows.append({'far': far, 'threshold': thr,
                     'fa_count_hist': cnt_hist,
                     'fa_count_pairs': cnt_pairs,
                     'expected_fpir_x_negtotal': far * total_neg})
        print(f"      FPIR {far:g} thr {thr:.3f}: "
              f"hist {cnt_hist:,} / 实捞 {cnt_pairs:,} "
              f"(期望 {far * total_neg:,.0f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'fa_counts.csv'),
                              index=False)

    summary = {
        'dataset': args.dataset,
        'ckpt_path': args.ckpt_path,
        'harvest_threshold': args.neg_threshold,
        'total_neg_pairs': total_neg,
        'total_pos_pairs': total_pos,
        'n_samples': int(n),
        'levels': rows,
    }

    if len(neg_pairs):
        dfp = pd.DataFrame(neg_pairs, columns=['i', 'j', 'score'])
        dfp = dfp.sort_values('score', ascending=False).reset_index(drop=True)
        dfp['i'] = dfp['i'].astype(np.int64)
        dfp['j'] = dfp['j'].astype(np.int64)
        paths = np.asarray(load_paths(args.kind, args.dataset))
        assert len(paths) == n, f'paths {len(paths)} != 样本 {n}'
        dfp['path_i'] = paths[dfp['i'].values]
        dfp['path_j'] = paths[dfp['j'].values]
        dfp['label_i'] = query_ids[dfp['i'].values]
        dfp['label_j'] = query_ids[dfp['j'].values]
        pq = os.path.join(args.out_dir, 'fa_pairs.parquet')
        dfp.to_parquet(pq, index=False)
        print(f"      fa_pairs.parquet: {len(dfp):,} 行", flush=True)

        # 震中图: 卷入 FA 对最多的图片 (撞脸簇锚点)
        both = pd.concat([dfp['i'], dfp['j']])
        vc = both.value_counts()
        summary['n_unique_images_in_fa'] = int(len(vc))
        top = vc.head(20)
        summary['top_offender_images'] = [
            {'index': int(k), 'n_fa_pairs': int(v),
             'path': str(paths[int(k)]), 'label': int(query_ids[int(k)])}
            for k, v in top.items()
        ]
        if args.sample_pairs > 0:
            n_exp = export_samples(args.kind, args.dataset, dfp,
                                   os.path.join(args.out_dir, 'sample_pairs'),
                                   args.sample_pairs)
            summary['sample_pairs_exported'] = int(n_exp)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 清理 /dev/shm 分片与 /tmp engine 临时目录
    shutil.rmtree(shm_path, ignore_errors=True)
    if os.path.exists('/tmp/trt_eval3_cache'):
        shutil.rmtree('/tmp/trt_eval3_cache')
    print(f"[4/4] 明细+汇总: {time.time()-t0:.1f}s -> {args.out_dir}", flush=True)
    print("FA 打捞完成!", flush=True)

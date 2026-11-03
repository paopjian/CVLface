"""Dataloader workers 数量扫描: 特征提取端到端吞吐 (单卡, fp16 engine)

背景: 端到端提特征 83% 时间在 CPU JPEG 解码 (engine 纯吞吐 7787 vs 端到端 644 img/s).
生产 7 卡 x num_workers=5 = 35 解码进程, 只占 128 核的 27%. 本实验扫描
num_workers 找单卡最优值, 生产通过 TRT_NUM_WORKERS 环境变量直接落地.

用法: python dataloader_bench.py
"""
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
import polars as pl
import torch

ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
ACC_DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'  # 稳态测试用大数据集
ENGINE_BATCH = 256
WORKER_LIST = (5, 10, 16, 24)
N_LIMIT = 120000  # 大样本: 降低 worker 启动开销占比


class DecodeDataset(torch.utils.data.Dataset):
    """decode + transform 在 __getitem__ (worker 进程) 执行, 与生产 HFIndexedDataset 同构"""

    def __init__(self, ds, tf):
        self.ds = ds
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]['image'].convert('RGB')
        return self.tf(img), idx


def _collate(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    indexes = torch.tensor([b[1] for b in batch])
    return pixel_values, indexes


def bench(nw, ds, tf):
    """单卡端到端提特征 (与生产 worker 同构): worker 并行 decode → flip 拼 batch → engine"""
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner  # noqa: 复用同款 runner
    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)
    subset = torch.utils.data.Subset(DecodeDataset(ds, tf),
                                     range(min(N_LIMIT, len(ds))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=ENGINE_BATCH // 2, num_workers=nw,
        collate_fn=_collate)

    t0 = time.time()
    for x, _ in loader:
        x = x.cuda()
        x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        runner(x_combined)
    dt = time.time() - t0
    del runner
    torch.cuda.empty_cache()
    return len(subset) / dt, dt


def main():
    torch.cuda.set_device(0)
    from datasets import load_from_disk
    from torchvision import transforms
    ds = load_from_disk(ACC_DATA)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    n = len(ds)
    print(f"数据集: {os.path.basename(ACC_DATA)} 全量 {n} 张, 取前 {min(N_LIMIT, n)} 张, engine=fp16 batch={ENGINE_BATCH}, CPU 核数={os.cpu_count()}")

    rows = []
    base = None
    for nw in WORKER_LIST:
        thr, dt = bench(nw, ds, tf)
        base = base or thr
        print(f"num_workers={nw:2d}: {thr:,.0f} img/s ({dt:.1f}s, "
              f"{thr/base:.2f}x vs nw=5)")
        rows.append({'num_workers': nw, 'img_per_sec': round(thr, 0),
                     'sec': round(dt, 2), 'speedup_vs_nw5': round(thr / base, 3)})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'dataloader_workers_results.parquet')
    pl.DataFrame(rows).write_parquet(out)
    print(f"结果已保存: {out}")


if __name__ == '__main__':
    main()

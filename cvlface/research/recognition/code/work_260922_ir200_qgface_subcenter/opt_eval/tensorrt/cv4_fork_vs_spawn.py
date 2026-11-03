"""cv4 对比: spawn(每 worker pickle 450MB dataset) vs fork(零 pickle)

用法: python cv4_fork_vs_spawn.py
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
import torch

ENGINE_BATCH = 256
N = 400000
ROOT = '/data1/dataset_0605/test_1201'


def _collate(batch):
    return torch.stack([b["pixel_values"] for b in batch])


def run(mp_ctx, nw=10):
    import cv2
    from eval_all_trt_single import FastImageFolderDataset
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner
    ds = FastImageFolderDataset(ROOT)
    subset = torch.utils.data.Subset(ds, range(min(N, len(ds))))
    t0 = time.time()
    loader = torch.utils.data.DataLoader(
        subset, batch_size=ENGINE_BATCH // 2, num_workers=nw,
        collate_fn=_collate, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
        multiprocessing_context=mp_ctx)
    runner = EngineRunner('/tmp/int8_ptq_work/engine_fp16.plan',
                          batch_size=ENGINE_BATCH)
    it = iter(loader)
    x0 = next(it)  # 首批 (含 worker 启动 + spawn pickle)
    t_first = time.time() - t0
    x = x0.cuda()
    runner(torch.cat([x, torch.flip(x, dims=[3])], dim=0))
    torch.cuda.synchronize()
    t1 = time.time()
    cnt = x0.shape[0]
    for xb in it:
        x = xb.cuda()
        runner(torch.cat([x, torch.flip(x, dims=[3])], dim=0))
        cnt += xb.shape[0]
    dt = time.time() - t1
    del loader, it, runner
    torch.cuda.empty_cache()
    return t_first, cnt, dt


def main():
    torch.cuda.set_device(0)
    for ctx in ['spawn', 'fork']:
        t_first, cnt, dt = run(ctx)
        print(f"{ctx:5s}: 首批等待 {t_first:.1f}s, 之后 {cnt:,} 张 / {dt:.1f}s "
              f"= {cnt/dt:,.0f} img/s")


if __name__ == '__main__':
    main()

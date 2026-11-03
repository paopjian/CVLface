"""方案 2 PoC: 每 GPU 2 个消费进程分片喂同一 engine

原理: 单进程端到端 3,646 img/s 的瓶颈是 python 供给与 CUDA sync 的交替停顿
(iter 35ms ≈ engine 34.5ms, 1:1 停顿). 两个进程各自 context + 各吃一半数据,
GPU 上 execute 排队串行, 但两进程的停顿互相错开 → GPU 间隙被填满.

用法: python two_consumer_poc.py [n_proc]  (默认 2)
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
import torch.multiprocessing as mp
import cv2

ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
ENGINE_BATCH = 256
DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
N_LIMIT = 120000


def _worker(rank, n_proc, result_q):
    import torch
    from datasets import load_from_disk
    from datasets.features.image import Image as HFImage
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner

    torch.cuda.set_device(0)
    ds = load_from_disk(DATA)
    ds_raw = ds.cast_column('image', HFImage(decode=False))

    class DecodeDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds_raw)
        def __getitem__(self, idx):
            raw = bytes(ds_raw[idx]['image']['bytes'])
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy((img - 0.5) / 0.5).permute(2, 0, 1)

    subset = DecodeDataset()
    # 手动交错分片 (免 DistributedSampler 的 dist 依赖)
    indices = list(range(min(N_LIMIT, len(subset))))[rank::n_proc]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(subset, indices), batch_size=ENGINE_BATCH // 2,
        num_workers=10, collate_fn=lambda b: torch.stack(b), pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
        multiprocessing_context='fork')  # CUDA 已初始化, worker 仅 CPU 解码, fork 安全

    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)
    for _ in range(3):  # 预热
        for xb in loader:
            x = xb.cuda()
            runner(torch.cat([x, torch.flip(x, dims=[3])], dim=0))
            break
    torch.cuda.synchronize()
    t0 = time.time()
    cnt = 0
    for xb in loader:
        x = xb.cuda()
        runner(torch.cat([x, torch.flip(x, dims=[3])], dim=0))
        cnt += xb.shape[0]
    dt = time.time() - t0
    result_q.put((rank, cnt, dt))


def main():
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.set_start_method('spawn', force=True)
    q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(r, n_proc, q)) for r in range(n_proc)]
    t0 = time.time()
    for p in procs:
        p.start()
    results = [q.get() for _ in range(n_proc)]
    for p in procs:
        p.join()
    wall = time.time() - t0
    total = sum(r[1] for r in results)
    print(f"进程数={n_proc}: 合计 {total:,} 张 / wall {wall:.1f}s = "
          f"{total/wall:,.0f} img/s  (单进程基线 3,644)")
    for rank, cnt, dt in sorted(results):
        print(f"  rank{rank}: {cnt:,} 张, {cnt/dt:,.0f} img/s (局部 {dt:.1f}s)")


if __name__ == '__main__':
    main()

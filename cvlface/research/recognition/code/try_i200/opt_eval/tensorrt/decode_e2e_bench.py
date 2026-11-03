"""解码加速端到端 bench: 最优 CPU 解码器 vs DALI GPU 解码, 目标逼近纯 engine 吞吐 (7,787 img/s)

单线程结论 (decode_bench.py): 生产 PIL 链 1054 img/s (30% 在 transforms),
tv_png 1718 最快且 PNG 无损位一致. 本脚本测端到端:
  A baseline    : 生产链路 (PIL + transforms, DecodeDataset) — 已知 nw=5 → 2,873 img/s
  B tv_png_cpu  : tv_png 解码 + 手工 /255 (免 transforms), nw 扫描
  C dali_gpu    : DALI GPU 解码 + GPU 归一化 + transpose, bytes 主线程预取供给

计时范围: 预取满足的前提下 decode → engine 全链路 (不含指标计算)
"""
import io
import os
import queue
import sys
import threading
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

from datasets import load_from_disk
from datasets.features.image import Image as HFImage

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
ENGINE_BATCH = 256
N_LIMIT = 120000


def get_raw_iter(ds_raw, n, q, stop, extra=0):
    """arrow bytes 供给线程"""
    for i in range(n + extra):
        if stop.is_set():
            return
        q.put(bytes(ds_raw[i % len(ds_raw)]['image']['bytes']))
    q.put(None)


# ---------------- A/B: torch DataLoader 路线 ----------------
class DecodeDataset(torch.utils.data.Dataset):
    """decoder: 'pil'(生产) / 'tv_png'(torchvision) / 'cv2'"""

    def __init__(self, ds, decoder='pil'):
        self.ds = ds
        self.decoder = decoder

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        raw = bytes(self.ds[idx]['image']['bytes']) \
            if isinstance(self.ds[idx]['image'], dict) else None
        if self.decoder == 'pil':
            from PIL import Image
            from torchvision import transforms as T
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            x = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])(img)
            return x
        if self.decoder == 'tv_png':
            from torchvision.io import decode_png
            t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8))
            img = decode_png(t).numpy().astype(np.float32) / 255.0
            return torch.from_numpy((img - 0.5) / 0.5)
        if self.decoder == 'cv2':
            import cv2
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy((img.transpose(2, 0, 1) - 0.5) / 0.5)
        raise ValueError(self.decoder)


def _collate(batch):
    return torch.stack(batch)


def bench_dataloader(ds_raw, decoder, nw, pin=False):
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner
    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)
    subset = torch.utils.data.Subset(
        DecodeDataset(ds_raw, decoder), range(min(N_LIMIT, len(ds_raw))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=ENGINE_BATCH // 2, num_workers=nw,
        collate_fn=_collate, pin_memory=pin)
    t0 = time.time()
    for x in loader:
        x = x.cuda()
        x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        runner(x_combined)
    dt = time.time() - t0
    del runner
    torch.cuda.empty_cache()
    return len(subset) / dt, dt


# ---------------- C: DALI GPU 解码 ----------------
class PreDecodedDataset(torch.utils.data.Dataset):
    """零解码定位: 预生成张量池直通, 保留真实 collate/H2D/engine 链"""

    def __init__(self, n, pool=4096):
        g = torch.Generator().manual_seed(0)
        self.pool = torch.rand(pool, 3, 112, 112, generator=g) * 2 - 1
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.pool[idx % len(self.pool)]


def bench_pretensor(nw):
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner
    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)
    subset = PreDecodedDataset(min(N_LIMIT, 10 ** 9))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=ENGINE_BATCH // 2, num_workers=nw, collate_fn=_collate)
    t0 = time.time()
    for x in loader:
        x = x.cuda()
        x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        runner(x_combined)
    dt = time.time() - t0
    del runner
    torch.cuda.empty_cache()
    return len(subset) / dt, dt


def bench_dali(ds_raw, dev='cpu'):
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner

    n = min(N_LIMIT, len(ds_raw))
    pipe = dali.pipeline.Pipeline(batch_size=ENGINE_BATCH, num_threads=4,
                                  device_id=0, prefetch_queue_depth=4)

    q = queue.Queue(maxsize=64)
    stop = threading.Event()
    feeder = threading.Thread(target=get_raw_iter,
                              args=(ds_raw, n, q, stop, 8 * ENGINE_BATCH),
                              daemon=True)

    with pipe:
        data = fn.external_source(device=dev, name='BYTES')  # gpu 时 feed 自动 H2D
        # 先在 CPU 解码 (PNG GPU 解码若可用再切 'gpu'), 统一 -1..1
        img = fn.decoders.image(data, device=dev, output_type=types.RGB)
        img = fn.cast(img, dtype=types.FLOAT) * (1 / 255.0)
        img = (img - 0.5) / 0.5
        img = fn.transpose(img, perm=[2, 0, 1])  # HWC → CHW
        pipe.set_outputs(img)

    pipe.build()
    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)

    def feed():
        batch = [q.get() for _ in range(ENGINE_BATCH)]
        pipe.feed_input('BYTES',
                        [np.frombuffer(b, np.uint8) for b in batch])

    try:
        feeder.start()
        # DALI feed_input 异步: 初始喂满 prefetch_queue_depth, 循环内先 run 腾位再 feed
        for _ in range(4):
            feed()
        for _ in range(2):
            pipe.run()
            feed()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n // ENGINE_BATCH):
            out = pipe.run()
            feed()
            x = torch.as_tensor(out[0].as_tensor())  # DALI → torch
            x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
            runner(x_combined)
        dt = time.time() - t0
    finally:
        stop.set()
    del runner
    torch.cuda.empty_cache()
    return n / dt, dt


def main():
    torch.cuda.set_device(0)
    ds = load_from_disk(DATA)
    ds_raw = ds.cast_column('image', HFImage(decode=False))
    n = min(N_LIMIT, len(ds_raw))
    print(f"数据: {os.path.basename(DATA)} 前 {n} 张, engine=fp16 bs={ENGINE_BATCH}")

    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    # 纯供给速率参照 (arrow bytes 读取)
    t0 = time.time()
    for i in range(5000):
        _ = ds_raw[i]['image']['bytes']
    supply = 5000 / (time.time() - t0)
    print(f"arrow bytes 纯供给 (单线程): {supply:,.0f} img/s")

    modes = [
        ('baseline_pil_nw5', lambda: bench_dataloader(ds_raw, 'pil', 5)),
        ('tv_png_nw5', lambda: bench_dataloader(ds_raw, 'tv_png', 5)),
        ('tv_png_nw10', lambda: bench_dataloader(ds_raw, 'tv_png', 10)),
        ('cv2_nw10', lambda: bench_dataloader(ds_raw, 'cv2', 10)),
        ('cv2_nw16', lambda: bench_dataloader(ds_raw, 'cv2', 16)),
        ('cv2_nw32', lambda: bench_dataloader(ds_raw, 'cv2', 32)),
        ('pretensor_nw8', lambda: bench_pretensor(8)),
        ('cv2_nw10_pin', lambda: bench_dataloader(ds_raw, 'cv2', 10, pin=True)),
        ('dali_cpu_decode', lambda: bench_dali(ds_raw, 'cpu')),
        ('dali_gpu_decode', lambda: bench_dali(ds_raw, 'gpu')),
    ]
    for name, fn_ in [m for m in modes if only is None or m[0] == only]:
        try:
            thr, dt = fn_()
            print(f"{name:18s}: {thr:,.0f} img/s ({dt:.1f}s)")
            rows.append({'mode': name, 'img_per_sec': round(thr, 0), 'sec': round(dt, 2)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{name:18s}: 失败 {type(e).__name__}: {str(e)[:150]}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'decode_e2e_results.parquet')
    if rows:
        pl.DataFrame(rows).write_parquet(out)
        print(f"结果已保存: {out}")


if __name__ == '__main__':
    main()

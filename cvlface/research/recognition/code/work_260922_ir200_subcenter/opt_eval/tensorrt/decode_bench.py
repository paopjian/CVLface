"""PNG 解码器对比 bench: 找提特征 CPU 解码瓶颈的加速方案

背景: 端到端提特征 83% 时间在解码 (HF arrow 内是 112x112 RGB PNG),
单 worker ~575 img/s, nw=5 已饱和 (2,873 img/s), 纯 engine 7,787 img/s.

对比:
  A pil        : PIL open + convert('RGB') + ToTensor + Normalize  (生产现状)
  B pil_norm   : PIL open + convert('RGB') + to_tensor/numpy/255   (免 torchvision 变换开销)
  C cv2        : cv2.imdecode + cvtColor BGR2RGB + /255
  D tv_png     : torchvision.io.decode_png + float/255
  E raw        : 零解码上限参照 (bytes 已经是解码后像素, 直接 reshape) — 需预处理

单线程速率 + 数值一致性 (max diff vs PIL)。端到端在 decode_e2e_bench 中测。
"""
import io
import os
import sys
import time

import numpy as np
import torch

from datasets import load_from_disk
from datasets.features.image import Image as HFImage

DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
N = 2000
REPEAT = 3


def get_raw_bytes(ds, n=N):
    """decode=False 拿原始 PNG bytes"""
    ds_raw = ds.cast_column('image', HFImage(decode=False))
    return [bytes(ds_raw[i]['image']['bytes']) for i in range(n)]


def to_float_chw(arr_u8):
    """HWC uint8 [0,255] → CHW float32 [0,1] (对齐 ToTensor 输出)"""
    return np.ascontiguousarray(arr_u8.transpose(2, 0, 1)).astype(np.float32) / 255.0


def bench_pil(raws):
    from torchvision import transforms as T
    from PIL import Image
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])  # 输出 [-1,1]

    def one(raw):
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        return tf(img).numpy()

    return one, '[-1,1]'


def bench_pil_norm(raws):
    from PIL import Image

    def one(raw):
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        return to_float_chw(np.asarray(img))

    return one, '[0,1]'


def bench_cv2(raws):
    import cv2

    def one(raw):
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return to_float_chw(img)

    return one, '[0,1]'


def bench_tv_png(raws):
    from torchvision.io import decode_png

    def one(raw):
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8))
        img = decode_png(t)  # CHW uint8 RGB
        return img.numpy().astype(np.float32) / 255.0

    return one, '[0,1]'


def main():
    torch.cuda.set_device(0)
    ds = load_from_disk(DATA)
    print(f"数据: {os.path.basename(DATA)}, 取前 {N} 张 PNG bytes")
    raws = get_raw_bytes(ds)
    print(f"平均 PNG 大小: {np.mean([len(r) for r in raws])/1024:.1f} KB")

    # PIL 基线输出 ([-1,1] 与 [0,1] 两种口径先算好)
    from PIL import Image
    from torchvision import transforms as T
    tf_norm = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    ref_01 = []
    for raw in raws[:200]:
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        ref_01.append(np.asarray(img))  # HWC uint8, 无损基准

    results = []
    for name, fn in [('pil', bench_pil), ('pil_norm', bench_pil_norm),
                     ('cv2', bench_cv2), ('tv_png', bench_tv_png)]:
        try:
            one, _ = fn(raws)
            # 正确性: 前 200 张与 uint8 基准比 (统一 [0,1])
            diffs = []
            for raw, ref in list(zip(raws, ref_01))[:200]:
                out = one(raw)  # [0,1] 或 [-1,1]
                out01 = (out + 1) / 2 if out.max() < 0 else out  # pil 是 [-1,1]
                # 更稳: 按解码器口径判断
                if out.min() < -0.01:
                    out01 = (out + 1) / 2
                diffs.append(np.abs(out01 - ref.transpose(2, 0, 1) / 255.0).max())
            max_diff = float(np.max(diffs))
            # 速率: 每轮 200 张 (避免 2000 张解码太慢的解码器), 取最好轮
            rates = []
            for _ in range(REPEAT):
                t0 = time.time()
                for raw in raws[:200]:
                    one(raw)
                rates.append(200 / (time.time() - t0))
            rate = max(rates)
            print(f"{name:9s}: {rate:7.0f} img/s/线程  max_diff={max_diff:.6f}")
            results.append({'decoder': name, 'img_per_sec_thread': round(rate, 1),
                            'max_diff_vs_pil_uint8': max_diff})
        except Exception as e:
            print(f"{name:9s}: 失败 {type(e).__name__}: {str(e)[:120]}")

    import polars as pl
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'decode_bench_results.parquet')
    pl.DataFrame(results).write_parquet(out)
    print(f"结果已保存: {out}")


if __name__ == '__main__':
    main()

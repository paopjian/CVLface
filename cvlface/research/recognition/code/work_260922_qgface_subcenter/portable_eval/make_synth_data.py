#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自造评测数据: 生成 ImageFolder 布局 + 同内容的 rec 打包, 供全链路演练.

数据特性: 同档案的图 = 同一基底图 + 微噪声 (近似重复图)。管线自检的硬信号
是 `--self-test` 闭合校验 (提取数=扫描数、对数闭合); TPIR 数值取决于模型
对噪声图的输出分布, 仅作参考。

用法:
  python make_synth_data.py --out /tmp/pe_data --sets "tiny:64:16,perf:2000:80"
      → /tmp/pe_data/folder/{tiny,perf}/{ARCHIVE_00000/xxx.jpg,...}
      → /tmp/pe_data/rec_{tiny,perf}/train.rec|train.idx|train.tsv|meta.json
  格式: 集名:档案数:每档案图数; 生成用 --workers 进程并行。

说明:
  - rec 与 mxnet RecordIO 兼容 (bundle_images_into_rec_v2 同格式):
    记录 = magic(4) + len|flag(4) + 头(flag i4/label f4/id Q8/id2 Q8) + jpeg + 4 对齐
  - cv2 编码 JPEG; 若无 cv2 退回 PIL
"""
import argparse
import io
import json
import multiprocessing as mp
import struct
from pathlib import Path

import numpy as np

MAGIC = 0xCED7230A


def encode_jpeg(img_bgr):
    try:
        import cv2
        ok, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        return enc.tobytes()
    except ImportError:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img_bgr[:, :, ::-1]).save(buf, 'JPEG', quality=90)
        return buf.getvalue()


def gen_archive_blobs(args):
    """worker: 生成一个档案的 jpeg blobs."""
    a, n_images, seed, noise = args
    rng = np.random.default_rng(seed + a)
    base = rng.integers(30, 225, (112, 112, 3), dtype=np.int16)
    blobs = []
    for _ in range(n_images):
        img = base + rng.normal(0, noise, (112, 112, 3)).astype(np.int16)
        img = np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1]  # → BGR
        blobs.append(encode_jpeg(img))
    return a, blobs


def gen_set(out_root, name, n_archives, imgs_per_archive, seed, workers):
    root = Path(out_root) / 'folder' / name
    tasks = [(a, imgs_per_archive, seed, 6) for a in range(n_archives)]
    blobs_by_a = {}
    if workers > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(workers) as pool:
            for a, blobs in pool.imap_unordered(gen_archive_blobs, tasks,
                                                chunksize=4):
                blobs_by_a[a] = blobs
                adir = root / f'ARCHIVE_{a:06d}'
                adir.mkdir(parents=True, exist_ok=True)
                for k, b in enumerate(blobs):
                    (adir / f'img_{k:05d}.jpg').write_bytes(b)
    else:
        for a, blobs in ((t[0], gen_archive_blobs(t)[1]) for t in tasks):
            blobs_by_a[a] = blobs
            adir = root / f'ARCHIVE_{a:06d}'
            adir.mkdir(parents=True, exist_ok=True)
            for k, b in enumerate(blobs):
                (adir / f'img_{k:05d}.jpg').write_bytes(b)
    n_total = n_archives * imgs_per_archive

    # rec 打包 (按档案序, 复用已编码 blobs)
    rec_dir = Path(out_root) / f'rec_{name}'
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec = open(rec_dir / 'train.rec', 'wb', buffering=1 << 20)
    idx = open(rec_dir / 'train.idx', 'w', buffering=1 << 20)
    tsv = open(rec_dir / 'train.tsv', 'w', buffering=1 << 20)
    offset = idx_i = 0
    for a in range(n_archives):
        for k, blob in enumerate(blobs_by_a[a]):
            data = struct.pack('<IfQQ', 0, float(a), idx_i, 0) + blob
            rec.write(struct.pack('<II', MAGIC, (0 << 29) | len(data)))
            rec.write(data)
            pad = (4 - len(data) % 4) % 4
            if pad:
                rec.write(b'\x00' * pad)
            idx.write(f'{idx_i}\t{offset}\n')
            tsv.write(f'{idx_i}\t{a}/img_{k:05d}.jpg\t{a}\n')
            offset += 8 + len(data) + pad
            idx_i += 1
    rec.close(); idx.close(); tsv.close()
    with open(rec_dir / 'meta.json', 'w') as f:
        json.dump({'num_classes': n_archives, 'num_samples': n_total}, f)
    return name, root, rec_dir, n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='/tmp/pe_data')
    ap.add_argument('--sets', default='tiny:64:16',
                    help='逗号分隔 "集名:档案数:每档案图数", 如 tiny:64:16,perf:2000:80')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=32,
                    help='生成并行进程数')
    args = ap.parse_args()

    for spec in args.sets.split(','):
        name, n_archives, n_imgs = spec.split(':')
        n_archives, n_imgs = int(n_archives), int(n_imgs)
        name, root, rec_dir, n_total = gen_set(
            args.out, name, n_archives, n_imgs, args.seed, args.workers)
        size_mb = (rec_dir / 'train.rec').stat().st_size / 1e6
        print(f'[{name}] {n_archives} 档案 × {n_imgs} 图 = {n_total:,} 张\n'
              f'  folder → {root}\n  rec    → {rec_dir} ({size_mb:.1f} MB)')
    example = args.sets.split(',')[0].split(':')[0]
    print('\n评估示例:')
    print(f'  python run_eval.py --engine <engine> --data {args.out}/folder/{example} '
          f'--source folder --pipeline nvjpeg --self-test')
    print(f'  python run_eval.py --engine <engine> --data {args.out}/rec_{example} '
          f'--source rec --pipeline nvjpeg --self-test')


if __name__ == '__main__':
    main()

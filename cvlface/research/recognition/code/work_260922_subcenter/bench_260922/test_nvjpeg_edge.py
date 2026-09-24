"""nvjpeg 满批约束与分片等长的边界验证。

验证点:
1. 尾批只有 1 张图 (257 = 1×256 + 1): 补满解码 + n_valid=1 截取, 不段错误
2. world=3、10 张图 (分片 4/3/3 不均): world 补齐后各 rank 等长 (4/4/4)
3. 极小数据 (3 张 < nv): 补到满批, n_valid=3
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluations.nvjpeg_pipeline import NvjpegFeeder  # noqa: E402


def make_folder(root, n):
    d = os.path.join(root, 'c0')
    os.makedirs(d)
    for i in range(n):
        img = np.full((112, 112, 3), (i * 7) % 255, np.uint8)
        cv2.imwrite(os.path.join(d, f'{i:03d}.jpg'), img)


def main():
    torch.cuda.set_device(0)
    tmp = tempfile.mkdtemp(prefix='pe_edge_')
    try:
        # 1. 尾批 1 张
        make_folder(tmp, 257)
        f = NvjpegFeeder('folder', tmp, 0, 1, 256, 0)
        assert len(f) == 2, f'{len(f)} 批'
        batches = list(f.batches(torch.cuda.current_stream()))
        torch.cuda.synchronize()
        assert [b.n_valid for b in batches] == [256, 1], \
            [b.n_valid for b in batches]
        assert batches[0].x.shape == (256, 3, 112, 112)
        print('[1] OK: 257 图 → 2 批 (256+1), 尾批补满解码后 n_valid=1, 无段错误')

        # 2. world=3, 10 图: 原 4/3/3 → 补齐 4/4/4
        tmp10 = tempfile.mkdtemp(prefix='pe_edge10_')
        try:
            make_folder(tmp10, 10)
            lens = []
            for r in range(3):
                fr = NvjpegFeeder('folder', tmp10, r, 3, 256, 0)
                assert fr.n_valid_total == 4, (r, fr.n_valid_total)  # 各 rank 等长
                b = next(iter(fr.batches(torch.cuda.current_stream())))
                lens.append((fr.n_valid_total, b.n_valid,
                             sorted(fr._shard_index[:fr.n_valid_total].tolist())))
            torch.cuda.synchronize()
            assert [l[0] for l in lens] == [4, 4, 4], lens
            assert [l[1] for l in lens] == [4, 4, 4], lens
            # 有效索引: 三 rank 并集 = 0..9 各一次 (world 补齐的 0,1 重复由 gather 去重)
            union = sorted(set(i for l in lens for i in l[2]))
            assert union == list(range(10)), union
            dup = [i for l in lens for i in l[2]]
            assert sorted(dup) == [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], sorted(dup)
            # 满批约束: nv 补齐后每 rank 批容量恰为 256 (decode 调用张数恒等)
            assert all(len(fr) == 1 for fr in
                       [NvjpegFeeder('folder', tmp10, r, 3, 256, 0) for r in range(3)])
            print('[2] OK: world=3×10 图 → 各 rank 等长 4, 有效索引并集恰覆盖 0..9 '
                  '(重复 0,1 由 gather 去重; nv 补齐只在批内, 不进有效索引)')
        finally:
            shutil.rmtree(tmp10, ignore_errors=True)

        # 2b. world=3, 257 图 (含尾批不均): 各 rank 等长 86
        lens = []
        for r in range(3):
            fr = NvjpegFeeder('folder', tmp, r, 3, 256, 0)
            b = next(iter(fr.batches(torch.cuda.current_stream())))
            lens.append((fr.n_valid_total, b.n_valid))
        torch.cuda.synchronize()
        assert [l[0] for l in lens] == [86, 86, 86], lens
        assert [l[1] for l in lens] == [86, 86, 86], lens
        print('[2b] OK: world=3×257 图 → 各 rank 等长 86 (原 86/86/85 → 补齐)')

        # 3. 极小: 3 张
        tmp2 = tempfile.mkdtemp(prefix='pe_edge2_')
        try:
            make_folder(tmp2, 3)
            f3 = NvjpegFeeder('folder', tmp2, 0, 1, 256, 0)
            assert len(f3) == 1
            b = next(iter(f3.batches(torch.cuda.current_stream())))
            torch.cuda.synchronize()
            assert b.n_valid == 3 and b.x.shape == (256, 3, 112, 112)
            print('[3] OK: 3 张图 → 补满 256 解码, n_valid=3')
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('EDGE ALL PASS')


if __name__ == '__main__':
    main()

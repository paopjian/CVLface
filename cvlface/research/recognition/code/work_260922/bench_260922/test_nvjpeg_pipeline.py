"""nvjpeg_pipeline 单元测试: 三种数据源解码正确性 + 坏图异常传播 (可捕获, 非段错误)。

单卡运行: CUDA_VISIBLE_DEVICES=0 python test_nvjpeg_pipeline.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations.nvjpeg_pipeline import NvjpegFeeder, detect_kind  # noqa: E402

VAL_GLINT = '/data1/dataset_260918/val_glint'
CPLFW_HF = '/data1/dataset_260918/facerec_val/cplfw'


def check_close(a, b, tag, tol_mean=5.0, tol_max=60):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    assert d.mean() < tol_mean and d.max() < tol_max, \
        f'{tag}: mean diff {d.mean():.2f} max {d.max()}'


def test_rec():
    assert detect_kind(VAL_GLINT) == 'rec'
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'portable_eval'))
    import eval_lib as el
    f = NvjpegFeeder('rec', VAL_GLINT, 0, 8, 256, 0)
    src = el.RecSource(VAL_GLINT)
    stream = torch.cuda.current_stream()
    n_checked = 0
    for b in f.batches(stream):
        x = b.x[:b.n_valid].float()  # (r,3,112,112) 已归一化
        # 反归一化回 uint8 与 cv2 路径逐张对比
        u8 = ((x * 0.5 + 0.5) * 255).round().clamp(0, 255).byte() \
            .permute(0, 2, 3, 1).cpu().numpy()
        idx = b.index[:8]
        ref = el.RecSource.read_records(src.fd, src.offsets[idx]).numpy()
        check_close(u8[:8], ref, f'rec batch {b.index[0]}')
        assert b.labels is not None and len(b.labels) == b.n_valid
        n_checked += b.n_valid
        if n_checked >= 512:
            break
    print(f'[rec] OK: 前 {n_checked} 张 nvjpeg vs cv2 像素差 < 阈值, '
          f'labels/index 对齐, 共 {len(f)} 批/分片')


def test_hf():
    assert detect_kind(CPLFW_HF) == 'hf'
    # cplfw 是 PNG: 应抛可捕获异常 (调用方据此降级 cv2), 而非 nvjpeg 报错/段错误
    try:
        NvjpegFeeder('hf', CPLFW_HF, 0, 8, 256, 0, with_paths=True)
        raise AssertionError('PNG 数据集未被拦截')
    except RuntimeError as e:
        assert '非 JPEG' in str(e)
    print('[hf] OK: PNG 数据集被 JPEG 魔数守卫拦截, 异常可捕获 (降级 cv2 语义)')


def test_folder_and_corrupt():
    import cv2
    tmp = tempfile.mkdtemp(prefix='pe_folder_test_')
    try:
        # 3 类 × 4 图, 内容渐变, 与 FolderSource 读取一致
        for ci in range(3):
            d = os.path.join(tmp, f'class_{ci}')
            os.makedirs(d)
            for j in range(4):
                img = np.full((112, 112, 3), 30 * ci + 20 * j, dtype=np.uint8)
                cv2.imwrite(os.path.join(d, f'{j}.jpg'), img)
        # 第 4 类放一张截断的坏 jpeg
        d = os.path.join(tmp, 'class_bad')
        os.makedirs(d)
        good = os.path.join(tmp, 'class_0', '0.jpg')
        raw = open(good, 'rb').read()
        open(os.path.join(d, 'bad.jpg'), 'wb').write(raw[:len(raw) // 2])

        assert detect_kind(tmp) == 'folder'
        f = NvjpegFeeder('folder', tmp, 0, 1, 4, 0)
        b = next(iter(f.batches(torch.cuda.current_stream())))
        assert b.labels is not None and b.n_valid == 4
        print('[folder] OK: 好图目录解码+labels 对齐')

        # 坏图: nvjpeg 应报错抛异常 (可被上层捕获降级), 而非段错误
        import shutil
        only_bad = os.path.join(tmp, 'only_bad', 'class_bad')  # 须按 class/ 子目录布局
        os.makedirs(only_bad)
        for k in range(4):  # 同一张坏图复制 4 份凑满批
            shutil.copy(os.path.join(tmp, 'class_bad', 'bad.jpg'),
                        os.path.join(only_bad, f'bad_{k}.jpg'))
        f2 = NvjpegFeeder('folder', os.path.dirname(only_bad), 0, 1, 4, 0)
        try:
            next(iter(f2.batches(torch.cuda.current_stream())))
            raise AssertionError('坏图未被检出')
        except RuntimeError as e:
            print(f'[corrupt] OK: 异常可捕获 ({e})')
        finally:
            shutil.rmtree(only_bad, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    test_rec()
    test_hf()
    test_folder_and_corrupt()
    print('ALL PASS')

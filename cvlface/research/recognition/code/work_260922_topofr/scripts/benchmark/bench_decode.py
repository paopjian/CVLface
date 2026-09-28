"""
JPEG 解码 + 归一化路径 Benchmark
对比 PIL / OpenCV / TurboJPEG / torchvision.io 的单线程解码速度

用法:
    conda run --no-capture-output -n cvlface python scripts/benchmark/bench_decode.py
"""
import io
import os
import time
import struct
import numpy as np
import torch
from PIL import Image

# ============ 配置 ============
REC_PATH = '/data1/dataset_0605/train_rec/train.rec'
IDX_PATH = '/data1/dataset_0605/train_rec/train.idx'
NUM_SAMPLES = 5000  # 解码样本数
WARMUP = 200


def load_jpeg_bytes_from_rec(num=NUM_SAMPLES + WARMUP):
    """从 RecordIO 中读取 raw JPEG bytes"""
    offsets = {}
    with open(IDX_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                offsets[int(parts[0])] = int(parts[1])

    keys = sorted(offsets.keys())
    # skip header (index 0)
    keys = keys[1:num + 1]

    jpeg_list = []
    label_list = []
    with open(REC_PATH, 'rb') as rec:
        for idx in keys:
            rec.seek(offsets[idx])
            header_bytes = rec.read(8)
            magic, lrecord = struct.unpack('<II', header_bytes)
            length = lrecord & ((1 << 29) - 1)
            data = rec.read(length)
            # unpack: flag(4B) + label(4B) + id(8B) + id2(8B) = 24B header
            flag = struct.unpack('<I', data[:4])[0]
            label_val = struct.unpack('<f', data[4:8])[0]
            if flag == 0:
                img_bytes = data[24:]
                label = int(label_val)
            else:
                label_end = 24 + flag * 4
                img_bytes = data[label_end:]
                label = int(struct.unpack('<f', data[24:28])[0])
            jpeg_list.append(img_bytes)
            label_list.append(label)

    return jpeg_list, label_list


# ============ 解码方案 ============

def decode_pil_current(jpeg_bytes):
    """方案1: 当前代码路径 (PIL → np → PIL → ToTensor → Normalize)"""
    img = Image.open(io.BytesIO(jpeg_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    sample = np.array(img)  # (H,W,3) uint8
    # 模拟 to_PIL + ToTensor + Normalize
    sample = torch.from_numpy(sample).permute(2, 0, 1).float() / 255.0
    sample = (sample - 0.5) / 0.5
    return sample


def decode_pil_direct(jpeg_bytes):
    """方案2: PIL 优化 (PIL → np → torch, 跳过多余转换)"""
    img = Image.open(io.BytesIO(jpeg_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.asarray(img)  # zero-copy view if possible
    sample = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    sample = (sample - 0.5) / 0.5
    return sample


def decode_cv2(jpeg_bytes):
    """方案3: OpenCV (cv2.imdecode → torch)"""
    import cv2
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, (H,W,3)
    # BGR→RGB + HWC→CHW + normalize, 一步到位
    sample = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    sample = (sample - 0.5) / 0.5
    return sample


def decode_cv2_inplace(jpeg_bytes):
    """方案3b: OpenCV 优化 (避免 BGR→RGB 拷贝)"""
    import cv2
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # in-place friendly
    sample = torch.from_numpy(img).permute(2, 0, 1).float()
    sample.div_(255.0).sub_(0.5).div_(0.5)  # in-place ops
    return sample


def decode_turbojpeg(jpeg_bytes):
    """方案4: TurboJPEG (libjpeg-turbo SIMD → torch)"""
    from turbojpeg import TurboJPEG, TJPF_RGB
    tj = TurboJPEG()
    img = tj.decode(jpeg_bytes, pixel_format=TJPF_RGB)  # RGB, (H,W,3) uint8
    sample = torch.from_numpy(img).permute(2, 0, 1).float()
    sample.div_(255.0).sub_(0.5).div_(0.5)
    return sample


def decode_turbojpeg_cached(jpeg_bytes, _tj=[None]):
    """方案4b: TurboJPEG 缓存实例"""
    from turbojpeg import TurboJPEG, TJPF_RGB
    if _tj[0] is None:
        _tj[0] = TurboJPEG()
    img = _tj[0].decode(jpeg_bytes, pixel_format=TJPF_RGB)
    sample = torch.from_numpy(img).permute(2, 0, 1).float()
    sample.div_(255.0).sub_(0.5).div_(0.5)
    return sample


def decode_torchvision_io(jpeg_bytes):
    """方案5: torchvision.io.decode_jpeg (C++ libjpeg-turbo 后端)"""
    import torchvision.io
    data = torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8)
    img = torchvision.io.decode_jpeg(data)  # (3, H, W) uint8
    sample = img.float()
    sample.div_(255.0).sub_(0.5).div_(0.5)
    return sample


# ============ Benchmark 主逻辑 ============

def bench_one(name, func, jpeg_list):
    """对单个解码方案计时"""
    # warmup
    for i in range(WARMUP):
        _ = func(jpeg_list[i])

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for i in range(WARMUP, WARMUP + NUM_SAMPLES):
        _ = func(jpeg_list[i])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    imgs_per_sec = NUM_SAMPLES / elapsed
    us_per_img = elapsed / NUM_SAMPLES * 1e6
    print(f'  {name:30s} | {imgs_per_sec:8.0f} imgs/s | {us_per_img:7.1f} us/img | {elapsed:.2f}s total')
    return imgs_per_sec


def verify_outputs(jpeg_bytes):
    """验证所有方案输出一致"""
    ref = decode_pil_current(jpeg_bytes)
    funcs = {
        'pil_direct': decode_pil_direct,
        'cv2': decode_cv2,
        'cv2_inplace': decode_cv2_inplace,
        'turbojpeg_cached': decode_turbojpeg_cached,
        'torchvision_io': decode_torchvision_io,
    }
    print('\n[验证] 输出一致性检查:')
    for name, func in funcs.items():
        out = func(jpeg_bytes)
        diff = (ref - out).abs().max().item()
        status = 'PASS' if diff < 0.02 else f'DIFF={diff:.4f}'
        print(f'  {name:30s} vs pil_current: max_diff={diff:.6f} [{status}]')


if __name__ == '__main__':
    print(f'加载 {NUM_SAMPLES + WARMUP} 条 JPEG 数据...')
    jpeg_list, label_list = load_jpeg_bytes_from_rec()
    print(f'  done. 平均 JPEG 大小: {np.mean([len(j) for j in jpeg_list]):.0f} bytes')

    # 验证
    verify_outputs(jpeg_list[0])

    # Benchmark
    print(f'\n[Benchmark] 解码 {NUM_SAMPLES} 张 112x112 JPEG (单线程):')
    print(f'  {"方案":30s} | {"速度":>8s}      | {"延迟":>7s}     | 总耗时')
    print(f'  {"-"*30}-+-{"-"*15}-+-{"-"*11}-+-{"-"*8}')

    results = {}
    results['1_pil_current'] = bench_one('1. PIL (当前路径)', decode_pil_current, jpeg_list)
    results['2_pil_direct'] = bench_one('2. PIL direct (优化转换)', decode_pil_direct, jpeg_list)
    results['3_cv2'] = bench_one('3. OpenCV', decode_cv2, jpeg_list)
    results['3b_cv2_inplace'] = bench_one('3b. OpenCV inplace', decode_cv2_inplace, jpeg_list)
    results['4_turbojpeg'] = bench_one('4. TurboJPEG (cached)', decode_turbojpeg_cached, jpeg_list)
    results['5_torchvision_io'] = bench_one('5. torchvision.io', decode_torchvision_io, jpeg_list)

    # 总结
    baseline = results['1_pil_current']
    print(f'\n[总结] 相对 PIL (当前) 的加速比:')
    for name, speed in sorted(results.items()):
        ratio = speed / baseline
        print(f'  {name:25s}: {ratio:.2f}x')

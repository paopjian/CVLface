"""
GPU 喂满率 Benchmark — 模拟真实训练循环, 测量 GPU 等待数据的时间占比

用法:
    conda run --no-capture-output -n cvlface python scripts/benchmark/bench_gpu_saturation.py

测试: DataLoader 吞吐 vs 模型前向+反向速度, 判断当前是否 IO bound
"""
import io
import os
import sys
import struct
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============ 配置 ============
REC_DIR = '/data1/dataset_0605/train_rec'
BATCH_SIZE = 256
NUM_BATCHES = 1000
WARMUP_BATCHES = 50
NUM_WORKERS = 8
DEVICE = 'cuda:0'
USE_AMP = True  # bf16-mixed 模拟实际训练


class RecDataset_Best(Dataset):
    """RecordIO + TurboJPEG (最快解码路径)"""

    def __init__(self, rec_dir):
        from dataset.recordio_reader import RecordIOReader
        idx_path = os.path.join(rec_dir, 'train.idx')
        rec_path = os.path.join(rec_dir, 'train.rec')
        self._rec_dir = rec_dir
        self._reader = None
        self._tj = None

        reader = RecordIOReader(idx_path, rec_path)
        data = reader.read_idx(0)
        header, _ = RecordIOReader.unpack(data)
        if header.flag > 0:
            self._length = int(header.label[0]) - 1
        else:
            self._length = len(reader.keys) - 1
        reader.close()

    def _get_reader(self):
        if self._reader is None:
            from dataset.recordio_reader import RecordIOReader
            idx_path = os.path.join(self._rec_dir, 'train.idx')
            rec_path = os.path.join(self._rec_dir, 'train.rec')
            self._reader = RecordIOReader(idx_path, rec_path)
        return self._reader

    def _get_tj(self):
        if self._tj is None:
            from turbojpeg import TurboJPEG
            self._tj = TurboJPEG()
        return self._tj

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        from turbojpeg import TJPF_RGB
        from dataset.recordio_reader import RecordIOReader as RR
        reader = self._get_reader()
        data = reader.read_idx(index + 1)
        header, img_bytes = RR.unpack(data)
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)

        tj = self._get_tj()
        img = tj.decode(img_bytes, pixel_format=TJPF_RGB)
        sample = torch.from_numpy(img).permute(2, 0, 1).float()
        sample.div_(255.0).sub_(0.5).div_(0.5)
        return sample, label


def bench_data_only(dataset, num_workers):
    """纯数据加载速度 (不含 GPU 计算)"""
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
        shuffle=True, pin_memory=True, drop_last=True,
        prefetch_factor=3, persistent_workers=num_workers > 0,
    )

    it = iter(loader)
    for _ in range(WARMUP_BATCHES):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    t0 = time.perf_counter()
    for _ in range(NUM_BATCHES):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
    elapsed = time.perf_counter() - t0
    return NUM_BATCHES * BATCH_SIZE / elapsed


def bench_compute_only(model, batch_size, device):
    """纯 GPU 前向+反向速度 (合成数据, bf16-mixed)"""
    model.train()
    dummy_input = torch.randn(batch_size, 3, 112, 112, device=device)

    # warmup
    for _ in range(20):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(dummy_input)
            loss = out.sum()
        loss.backward()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_BATCHES):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(dummy_input)
            loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return NUM_BATCHES * batch_size / elapsed


def bench_full_loop(dataset, model, num_workers, device):
    """完整训练循环 (数据加载 + GPU 前向反向)"""
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
        shuffle=True, pin_memory=True, drop_last=True,
        prefetch_factor=3, persistent_workers=num_workers > 0,
    )
    model.train()

    it = iter(loader)
    for _ in range(WARMUP_BATCHES):
        try:
            imgs, labels = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, labels = next(it)
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(imgs)
            loss = out.sum()
        loss.backward()

    torch.cuda.synchronize()
    data_wait_time = 0.0
    t0 = time.perf_counter()
    for _ in range(NUM_BATCHES):
        td0 = time.perf_counter()
        try:
            imgs, labels = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, labels = next(it)
        imgs = imgs.to(device, non_blocking=True)
        td1 = time.perf_counter()
        data_wait_time += (td1 - td0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(imgs)
            loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    total_imgs = NUM_BATCHES * BATCH_SIZE
    return {
        'total_imgs_per_sec': total_imgs / total_time,
        'data_wait_pct': data_wait_time / total_time * 100,
        'total_time': total_time,
        'data_wait_time': data_wait_time,
    }


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, work_dir)

    print('='*80)
    print(f'GPU 喂满率 Benchmark')
    print(f'  batch_size={BATCH_SIZE}, num_batches={NUM_BATCHES}, device={DEVICE}')
    print('='*80)

    # 加载模型 (IR101)
    print('\n加载 IR101 模型...')
    from models.iresnet.model import IR_101
    model = IR_101(input_size=(112, 112), output_dim=512).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    print(f'  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # 加载数据集
    print('\n初始化 Dataset (RecordIO+TurboJPEG)...')
    dataset = RecDataset_Best(REC_DIR)
    print(f'  samples: {len(dataset):,}')

    # 1. 纯数据吞吐
    print(f'\n[1] 纯数据加载速度 (num_workers={NUM_WORKERS}):')
    data_speed = bench_data_only(dataset, NUM_WORKERS)
    print(f'  {data_speed:.0f} imgs/s')

    # 2. 纯 GPU 计算速度
    print(f'\n[2] 纯 GPU 前向+反向速度 (合成数据):')
    compute_speed = bench_compute_only(model, BATCH_SIZE, DEVICE)
    print(f'  {compute_speed:.0f} imgs/s')

    # 3. 完整循环
    print(f'\n[3] 完整训练循环 (数据 + 计算):')
    result = bench_full_loop(dataset, model, NUM_WORKERS, DEVICE)
    print(f'  总吞吐: {result["total_imgs_per_sec"]:.0f} imgs/s')
    print(f'  数据等待占比: {result["data_wait_pct"]:.1f}%')
    print(f'  总耗时: {result["total_time"]:.1f}s, 数据等待: {result["data_wait_time"]:.1f}s')

    # 4. 判断瓶颈
    print(f'\n[结论]')
    if result["data_wait_pct"] > 30:
        print(f'  IO BOUND! 数据等待 {result["data_wait_pct"]:.1f}% 时间')
        print(f'  建议: 增加 num_workers, 或换更快的解码器')
        theoretical_max = compute_speed
        current = result["total_imgs_per_sec"]
        headroom = (theoretical_max - current) / current * 100
        print(f'  理论加速空间: {headroom:.0f}% (compute={compute_speed:.0f} vs actual={current:.0f})')
    else:
        print(f'  COMPUTE BOUND. 数据等待仅 {result["data_wait_pct"]:.1f}%')
        print(f'  GPU 已接近饱和, 数据加载不是瓶颈')
        print(f'  进一步加速需要: torch.compile / 更大 batch / 多卡')

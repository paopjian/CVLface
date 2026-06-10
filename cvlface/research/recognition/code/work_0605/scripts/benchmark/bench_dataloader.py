"""
DataLoader 端到端 Benchmark — 对比不同解码后端 + DataLoader 配置
模拟真实训练的数据加载吞吐 (不含模型前向)

用法:
    conda run --no-capture-output -n cvlface python scripts/benchmark/bench_dataloader.py

评估: 1000 batch, batch_size=512, 模拟 7 GPU 中单卡的加载速度
"""
import io
import os
import sys
import struct
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ============ 配置 ============
REC_DIR = '/data1/dataset_0605/train_rec'
LMDB_DIR = '/data1/dataset_0605/train_lmdb/train.lmdb'
BATCH_SIZE = 512
NUM_BATCHES = 1000
NUM_WORKERS_LIST = [8, 12, 16]
WARMUP_BATCHES = 50

# ============ Dataset 定义 ============


class RecDataset_PIL(Dataset):
    """当前路径: RecordIO + PIL 解码"""

    def __init__(self, rec_dir):
        from dataset.recordio_reader import RecordIOReader
        idx_path = os.path.join(rec_dir, 'train.idx')
        rec_path = os.path.join(rec_dir, 'train.rec')
        self._rec_dir = rec_dir
        self._reader = None

        # read header for length
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

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        from dataset.recordio_reader import RecordIOReader as RR
        reader = self._get_reader()
        data = reader.read_idx(index + 1)
        header, img_bytes = RR.unpack(data)
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)

        # PIL 解码 (当前路径)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        sample = np.array(img)
        sample = torch.from_numpy(sample).permute(2, 0, 1).float() / 255.0
        sample = (sample - 0.5) / 0.5
        return sample, label


class RecDataset_CV2(Dataset):
    """RecordIO + OpenCV 解码"""

    def __init__(self, rec_dir):
        from dataset.recordio_reader import RecordIOReader
        idx_path = os.path.join(rec_dir, 'train.idx')
        rec_path = os.path.join(rec_dir, 'train.rec')
        self._rec_dir = rec_dir
        self._reader = None

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

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        import cv2
        from dataset.recordio_reader import RecordIOReader as RR
        reader = self._get_reader()
        data = reader.read_idx(index + 1)
        header, img_bytes = RR.unpack(data)
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)

        # OpenCV 解码
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = torch.from_numpy(img).permute(2, 0, 1).float()
        sample.div_(255.0).sub_(0.5).div_(0.5)
        return sample, label


class RecDataset_TurboJPEG(Dataset):
    """RecordIO + TurboJPEG 解码"""

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

        # TurboJPEG 解码
        tj = self._get_tj()
        img = tj.decode(img_bytes, pixel_format=TJPF_RGB)  # (H,W,3) uint8
        sample = torch.from_numpy(img).permute(2, 0, 1).float()
        sample.div_(255.0).sub_(0.5).div_(0.5)
        return sample, label


class RecDataset_TorchvisionIO(Dataset):
    """RecordIO + torchvision.io.decode_jpeg 解码"""

    def __init__(self, rec_dir):
        from dataset.recordio_reader import RecordIOReader
        idx_path = os.path.join(rec_dir, 'train.idx')
        rec_path = os.path.join(rec_dir, 'train.rec')
        self._rec_dir = rec_dir
        self._reader = None

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

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        import torchvision.io
        from dataset.recordio_reader import RecordIOReader as RR
        reader = self._get_reader()
        data = reader.read_idx(index + 1)
        header, img_bytes = RR.unpack(data)
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)

        # torchvision.io 解码 (C++ libjpeg-turbo)
        data_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
        img = torchvision.io.decode_jpeg(data_tensor)  # (3,H,W) uint8
        sample = img.float()
        sample.div_(255.0).sub_(0.5).div_(0.5)
        return sample, label


class LMDBDataset_TurboJPEG(Dataset):
    """LMDB + TurboJPEG 解码"""

    def __init__(self, lmdb_dir):
        import lmdb as _lmdb
        self._lmdb_dir = lmdb_dir
        self._env = None
        self._tj = None

        env = _lmdb.open(lmdb_dir, readonly=True, lock=False,
                         readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self._length = int(txn.get(b'__len__').decode())
        env.close()

    def _get_env(self):
        if self._env is None:
            import lmdb as _lmdb
            self._env = _lmdb.open(self._lmdb_dir, readonly=True, lock=False,
                                   readahead=False, meminit=False)
        return self._env

    def _get_tj(self):
        if self._tj is None:
            from turbojpeg import TurboJPEG
            self._tj = TurboJPEG()
        return self._tj

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        from turbojpeg import TJPF_RGB
        env = self._get_env()
        key = f'{index:08d}'.encode('ascii')
        with env.begin(write=False) as txn:
            raw = txn.get(key)
        label = struct.unpack('<i', raw[:4])[0]

        tj = self._get_tj()
        img = tj.decode(raw[4:], pixel_format=TJPF_RGB)
        sample = torch.from_numpy(img).permute(2, 0, 1).float()
        sample.div_(255.0).sub_(0.5).div_(0.5)
        return sample, label


# ============ Benchmark 逻辑 ============

def bench_dataloader(dataset, num_workers, persistent=False, tag=''):
    """测量 DataLoader 吞吐"""
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3 if num_workers > 0 else None,
        persistent_workers=persistent and num_workers > 0,
    )

    # warmup
    it = iter(loader)
    for _ in range(WARMUP_BATCHES):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

    # timed run
    total_imgs = 0
    t0 = time.perf_counter()
    for i in range(NUM_BATCHES):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        total_imgs += batch[0].shape[0]
    elapsed = time.perf_counter() - t0

    imgs_per_sec = total_imgs / elapsed
    batch_per_sec = NUM_BATCHES / elapsed
    print(f'  {tag:40s} | workers={num_workers:2d} | persist={str(persistent):5s} | '
          f'{imgs_per_sec:8.0f} imgs/s | {batch_per_sec:.1f} batch/s | {elapsed:.1f}s')
    return imgs_per_sec


if __name__ == '__main__':
    # 确保能 import dataset 模块
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, work_dir)

    print('='*100)
    print(f'DataLoader Benchmark: {NUM_BATCHES} batches x {BATCH_SIZE} = {NUM_BATCHES*BATCH_SIZE:,} images')
    print(f'warmup: {WARMUP_BATCHES} batches')
    print('='*100)

    # 构建 datasets
    datasets = {}
    print('\n初始化 datasets...')
    datasets['RecordIO+PIL'] = RecDataset_PIL(REC_DIR)
    datasets['RecordIO+CV2'] = RecDataset_CV2(REC_DIR)
    datasets['RecordIO+TurboJPEG'] = RecDataset_TurboJPEG(REC_DIR)
    datasets['RecordIO+torchvision.io'] = RecDataset_TorchvisionIO(REC_DIR)
    if os.path.isdir(LMDB_DIR):
        datasets['LMDB+TurboJPEG'] = LMDBDataset_TurboJPEG(LMDB_DIR)
    print(f'  {len(datasets)} datasets ready')

    # Benchmark 各组合
    results = {}
    for nw in NUM_WORKERS_LIST:
        print(f'\n--- num_workers={nw} ---')
        for name, ds in datasets.items():
            tag = f'{name}'
            speed = bench_dataloader(ds, num_workers=nw, persistent=False, tag=tag)
            results[(name, nw, False)] = speed

    # persistent_workers 对比 (用最优解码器)
    print(f'\n--- persistent_workers=True 对比 ---')
    best_ds_name = 'RecordIO+TurboJPEG'
    best_ds = datasets[best_ds_name]
    for nw in NUM_WORKERS_LIST:
        tag = f'{best_ds_name} (persistent)'
        speed = bench_dataloader(best_ds, num_workers=nw, persistent=True, tag=tag)
        results[(best_ds_name + '_persist', nw, True)] = speed

    # 总结
    print('\n' + '='*100)
    print('[总结] 各方案吞吐 (imgs/s):')
    print(f'  {"方案":35s} | {"w=8":>10s} | {"w=12":>10s} | {"w=16":>10s}')
    print(f'  {"-"*35}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}')
    for name in datasets.keys():
        speeds = [results.get((name, nw, False), 0) for nw in NUM_WORKERS_LIST]
        print(f'  {name:35s} | {speeds[0]:10.0f} | {speeds[1]:10.0f} | {speeds[2]:10.0f}')

    # persistent
    speeds_p = [results.get((best_ds_name + '_persist', nw, True), 0) for nw in NUM_WORKERS_LIST]
    print(f'  {best_ds_name + " (persistent)":35s} | {speeds_p[0]:10.0f} | {speeds_p[1]:10.0f} | {speeds_p[2]:10.0f}')

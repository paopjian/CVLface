"""nvjpeg GPU 批量解码管线 (基于 portable_eval/eval_lib), 供 eval_all_trt_single /
CustomVerificationEvaluator 共用。

关键约束 (portable_eval README 2026-09-20 溯源结论, load-bearing):
  - nvjpegDecodeBatched 每次调用的张数必须恰好等于 pe_nvjpeg_init 的 batch,
    部分批会段错误 (无法被 try/except 捕获)。本模块一律把尾批用末元素补满,
    通过 n_valid 告知调用方截取, 因此永不发起部分批调用。
  - 每次 init 的 batch 上限 256 (512×大JPEG 会触发库内部限制), batch>256 时
    自动按 256 分片多次调用。

任何异常 (import/编译/init/解码) 均向上抛出, 由调用方捕获后降级 cv2 DataLoader。
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PORTABLE_DIR = Path(__file__).resolve().parent.parent / 'portable_eval'


def _eval_lib():
    if str(_PORTABLE_DIR) not in sys.path:
        sys.path.insert(0, str(_PORTABLE_DIR))
    import eval_lib as el
    return el


class Batch:
    """一个解码批: x 已按 (x/255-0.5)/0.5 归一化到 fp16 BCHW (在 decode stream 上)。"""

    __slots__ = ('x', 'labels', 'index', 'paths', 'n_valid', 'event', 'keep')

    def __init__(self, x, labels, index, paths, n_valid, event, keep):
        self.x = x
        self.labels = labels
        self.index = index
        self.paths = paths
        self.n_valid = n_valid
        self.event = event
        self.keep = keep  # 持有 host blob 引用, 解码完成前不可释放


class NvjpegFeeder:
    """按 rank::world 分片逐批 GPU 解码。

    kind:
      'rec'    — dataset 目录含 train.rec/train.idx/train.tsv (标签取 tsv 第 3 列)
      'folder' — ImageFolder 布局 (类别=排序后目录名, 与 torchvision.ImageFolder 一致)
      'hf'     — HuggingFace Dataset (arrow), image 列以 bytes 读取; with_paths=True 时带出 path
    用法:
      feeder = NvjpegFeeder(kind, path, rank, world, batch, device, with_paths)
      for b in feeder.batches(decode_stream):
          torch.cuda.current_stream().wait_event(b.event)  # 消费前等待
          use b.x[:b.n_valid], b.labels, b.index, b.paths
    """

    NV_CAP = 256

    def __init__(self, kind, path, rank, world, batch, device, with_paths=False):
        assert kind in ('rec', 'folder', 'hf')
        self.kind = kind
        self.path = str(path)
        self.rank = rank
        self.world = world
        self.batch = int(batch)
        self.device = int(device)
        self.with_paths = with_paths
        self.nv = min(self.batch, self.NV_CAP)
        self._slot = 0

        el = _eval_lib()
        self._el = el
        self._ext = el.build_nvjpeg_ext()
        self._ext.pe_nvjpeg_init(self.device, self.nv)

        if kind == 'rec':
            src = el.RecSource(self.path)
            self._fd = src.fd
            items = src.offsets
            self._labels_all = src.ids
            self._get = self._get_rec
        elif kind == 'folder':
            src = el.FolderSource(self.path)
            items = np.array(src.paths, dtype=object)
            self._labels_all = np.asarray(src.ids, dtype=np.int64)
            self._fd = -1
            self._get = self._get_folder
        else:
            from datasets import Dataset as HFDataset, Image as HFImage
            self._ds = HFDataset.load_from_disk(self.path) \
                .cast_column('image', HFImage(decode=False))
            items = np.arange(len(self._ds), dtype=np.int64)
            self._labels_all = None
            self._fd = -1
            self._get = self._get_hf
            # nvjpeg 仅支持 JPEG: 首张魔数探测, PNG 等直接抛错由调用方降级 cv2
            head = self._ds[0]['image']['bytes'][:2]
            if head != b'\xff\xd8':
                raise RuntimeError(
                    f'hf 数据集 {self.path} 非 JPEG (首张魔数 {head.hex()}), '
                    f'nvjpeg 仅支持 JPEG, 请用 cv2 路径')

        self.n_total = len(items)
        # 分片前先把全局索引补齐到 world 整除 (重复开头索引, 与 DistributedSampler
        # 语义一致): torch 路径的 gather 用 torch.stack 要求各 rank 等长; 重复索引
        # 由 gather 的去重 (remove_duplicates / deduplicate) 吸收
        gidx = np.arange(self.n_total, dtype=np.int64)
        wpad = (-self.n_total) % self.world
        if wpad:
            gidx = np.concatenate([gidx, np.arange(wpad, dtype=np.int64)])
        shard_pos = gidx[self.rank::self.world]  # 各 rank 等长
        self._shard = items[shard_pos]
        self._shard_index = shard_pos            # 全局行号 (含补齐重复)
        self._shard_labels = self._labels_all[shard_pos] \
            if self._labels_all is not None else None
        # 补满到 nv 整数倍: 永不发起部分批调用
        pad = (-len(self._shard)) % self.nv
        if pad:
            self._shard = np.concatenate([self._shard, np.repeat(self._shard[-1:], pad)])
            self._shard_index = np.concatenate(
                [self._shard_index, np.repeat(self._shard_index[-1:], pad)])
            if self._shard_labels is not None:
                self._shard_labels = np.concatenate(
                    [self._shard_labels, np.repeat(self._shard_labels[-1:], pad)])
        self.n_batches = len(self._shard) // self.nv
        self.n_valid_total = len(self._shard) - pad

    def __len__(self):
        return self.n_batches

    # ---- 三种数据源的批读取 (返回喂给 pe_decode_* 的原料) ----

    def _get_rec(self, piece):
        offs = torch.from_numpy(np.asarray(piece, dtype=np.int64))
        return ('rec', offs)

    def _get_folder(self, piece):
        buf, offs, sizes = self._el.FolderSource.read_blobs(list(piece))
        return ('host', buf, offs, sizes)

    def _get_hf(self, piece):
        blobs, paths = [], []
        for i in piece:
            item = self._ds[int(i)]
            b = item['image']['bytes']
            if b[:2] != b'\xff\xd8':
                raise RuntimeError(f'hf 第 {i} 张非 JPEG, nvjpeg 仅支持 JPEG')
            blobs.append(b)
            if self.with_paths:
                p = item.get('path', '') or item['image'].get('path', '')
                paths.append(p)
        import numpy as _np
        sizes = _np.array([len(b) for b in blobs], dtype=np.int64)
        offs = _np.concatenate([[0], _np.cumsum(sizes)[:-1]]).astype(_np.int64)
        buf = torch.from_numpy(_np.frombuffer(b''.join(blobs), dtype=_np.uint8).copy())
        return ('host', buf, torch.from_numpy(offs), torch.from_numpy(sizes), paths)

    def batches(self, stream):
        """生成器: 在 stream 上发起解码+归一化并记录 event, 逐批 yield Batch。

        u8/x 双 slot 轮转, 支持调用方超前 next() 一次实现解码/推理重叠。
        """
        dev = f'cuda:{self.device}'
        u8 = [torch.empty(self.nv, 112, 112, 3, dtype=torch.uint8, device=dev)
              for _ in range(2)]
        x16 = [torch.empty(self.nv, 3, 112, 112, dtype=torch.float16, device=dev)
               for _ in range(2)]

        for bi in range(self.n_batches):
            sl = slice(bi * self.nv, (bi + 1) * self.nv)
            piece = self._shard[sl]
            raw = self._get(piece)
            slot = self._slot
            self._slot ^= 1
            keep = raw  # host blob 生存期绑定到本批
            with torch.cuda.stream(stream):
                if raw[0] == 'rec':
                    self._ext.pe_decode_rec(self._fd, raw[1], u8[slot], 24, slot,
                                            stream.cuda_stream, 0)
                else:
                    self._ext.pe_decode_host(raw[1], raw[2], raw[3], u8[slot], slot,
                                             stream.cuda_stream, 0)
                x = u8[slot].permute(0, 3, 1, 2).float() \
                    .div_(255.0).sub_(0.5).div_(0.5).half()
                x16[slot].copy_(x)
            ev = torch.cuda.Event()
            ev.record(stream)
            n_valid = min(self.nv, self.n_valid_total - bi * self.nv)
            labels = None
            if self._shard_labels is not None:
                labels = self._shard_labels[sl][:n_valid] if n_valid < self.nv \
                    else self._shard_labels[sl]
            index = self._shard_index[sl][:n_valid]
            paths = None
            if self.kind == 'hf' and self.with_paths:
                paths = raw[4][:n_valid]
            yield Batch(x16[slot], labels, index, paths, n_valid, ev, keep)


def detect_kind(path):
    """按目录内容判定数据源类型: rec / folder / hf。"""
    p = Path(path)
    if (p / 'train.rec').exists() and (p / 'train.idx').exists():
        return 'rec'
    if (p / 'state.json').exists() or (p / 'dataset_info.json').exists():
        return 'hf'
    return 'folder'


def available():
    """nvjpeg 管线是否可用 (仅做轻量检查: 依赖发现; 编译失败在首次使用时抛出)。"""
    try:
        el = _eval_lib()
        el.find_nvjpeg()
        el.find_cuda_include()
        return True
    except Exception:
        return False

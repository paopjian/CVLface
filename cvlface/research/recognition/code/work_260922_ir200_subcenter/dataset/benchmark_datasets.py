"""Benchmark Dataset classes for 6 data formats."""
import io
import os
import struct
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageFolderFaceDataset(Dataset):
    """Standard ImageFolder-style dataset (label dirs with image files)."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            idx = self.class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                path = os.path.join(cls_dir, fname)
                if os.path.isfile(path):
                    self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class LMDBFaceDataset(Dataset):
    """Read LMDB created by bundle_lmdb.py."""

    def __init__(self, lmdb_dir, transform=None):
        import lmdb as _lmdb
        self.transform = transform

        lmdb_path = os.path.join(lmdb_dir, 'train.lmdb')
        if not os.path.isdir(lmdb_path):
            lmdb_path = lmdb_dir

        self.env = _lmdb.open(lmdb_path, readonly=True, lock=False,
                              readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self._length = int(txn.get(b'__len__').decode())

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        key = f'{index:08d}'.encode('ascii')
        with self.env.begin(write=False) as txn:
            raw = txn.get(key)
        label = struct.unpack('<i', raw[:4])[0]
        img = Image.open(io.BytesIO(raw[4:])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class RecordIOFaceDataset(Dataset):
    """Read RecordIO using pure Python reader."""

    def __init__(self, rec_dir, transform=None):
        from dataset.recordio_reader import RecordIOReader
        self.transform = transform
        self.rec_dir = rec_dir

        idx_path = os.path.join(rec_dir, 'train.idx')
        rec_path = os.path.join(rec_dir, 'train.rec')
        reader = RecordIOReader(idx_path, rec_path)

        # Read header (index 0) to determine sample range
        data = reader.read_idx(0)
        header, _ = RecordIOReader.unpack(data)
        if header.flag > 0:
            self._length = int(header.label[0]) - 1
            self._start_idx = 1
        else:
            self._length = len(reader.keys) - 1
            self._start_idx = 1
        reader.close()
        self._reader = None

    def _get_reader(self):
        """Lazy-open reader per worker process (avoids file handle sharing after fork)."""
        if self._reader is None:
            from dataset.recordio_reader import RecordIOReader
            idx_path = os.path.join(self.rec_dir, 'train.idx')
            rec_path = os.path.join(self.rec_dir, 'train.rec')
            self._reader = RecordIOReader(idx_path, rec_path)
        return self._reader

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        from dataset.recordio_reader import RecordIOReader as RR
        reader = self._get_reader()
        data = reader.read_idx(index + self._start_idx)
        header, img_bytes = RR.unpack(data)
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class WebDatasetFaceDataset(torch.utils.data.IterableDataset):
    """Read WebDataset tar shards as a streaming IterableDataset.

    This is the native usage pattern for webdataset — streaming from tar shards
    with a shuffle buffer, reflecting how it would be used in real training.
    """

    def __init__(self, wds_dir, transform=None, shuffle_buffer=1000):
        import webdataset as wds_lib
        import glob
        import json as _json

        super().__init__()
        self.transform = transform
        self.wds_dir = wds_dir

        # Find all shards
        shard_pattern = os.path.join(wds_dir, 'shard-*.tar')
        self.shards = sorted(glob.glob(shard_pattern))
        self.shuffle_buffer = shuffle_buffer

        # Read metadata for length
        meta_path = os.path.join(wds_dir, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
            self._length = meta['num_samples']
        else:
            self._length = 0

    def __len__(self):
        return self._length

    def __iter__(self):
        import webdataset as wds_lib

        dataset = (
            wds_lib.WebDataset(self.shards, shardshuffle=100, empty_check=False,
                               nodesplitter=wds_lib.split_by_node)
            .shuffle(self.shuffle_buffer)
            .decode('pil')
            .to_tuple('jpg;png', 'cls')
        )

        for img, cls_bytes in dataset:
            label = int(cls_bytes)
            if self.transform:
                img = self.transform(img)
            yield img, label


class HDF5FaceDataset(Dataset):
    """Read HDF5 created by bundle_hdf5.py."""

    def __init__(self, h5_dir, transform=None):
        import h5py
        self.transform = transform
        self.h5_path = os.path.join(h5_dir, 'train.h5')
        # Only read metadata here, don't keep file open (fork-safety)
        with h5py.File(self.h5_path, 'r') as f:
            self._length = int(f.attrs['num_samples'])
        self._h5_file = None

    def _get_h5(self):
        """Lazy-open per worker process (avoids file handle sharing after fork)."""
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        h5 = self._get_h5()
        img_bytes = h5['images'][index].tobytes()
        label = int(h5['labels'][index])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ZarrFaceDataset(Dataset):
    """Read Zarr created by bundle_zarr.py using tensorstore (fast async IO)."""

    def __init__(self, zarr_dir, transform=None):
        self.transform = transform
        self.zarr_path = os.path.join(zarr_dir, 'train.zarr')

        # Read num_samples from zarr metadata (lightweight, no threading)
        import zarr as _zarr
        store = _zarr.open_group(self.zarr_path, mode='r')
        self._length = int(store.attrs['num_samples'])
        # Don't open tensorstore here — it uses internal threads incompatible with fork
        self._images = None
        self._labels = None

    def _get_stores(self):
        """Lazy-open tensorstore per worker process (fork-safe)."""
        if self._images is None:
            import tensorstore as ts
            self._images = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': self.zarr_path + '/images'},
            }).result()
            self._labels = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': self.zarr_path + '/labels'},
            }).result()
        return self._images, self._labels

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        images, labels = self._get_stores()
        img_array = images[index].read().result()
        label = int(labels[index].read().result())
        img = Image.fromarray(np.array(img_array))
        if self.transform:
            img = self.transform(img)
        return img, label


class StreamingFaceDataset:
    """Read MosaicML Streaming (MDS) dataset.

    Directly uses streaming.StreamingDataset (which is an IterableDataset).
    Handles distributed sharding internally — no DistributedSampler needed.
    """

    def __new__(cls, streaming_dir, transform=None, shuffle=False, batch_size=256):
        """Return a streaming.StreamingDataset subclass instance with transform applied."""
        from streaming import StreamingDataset as _StreamingDataset

        class _TransformedStreamingDataset(_StreamingDataset):
            def __init__(self, transform, **kwargs):
                super().__init__(**kwargs)
                self._transform = transform

            def __getitem__(self, index):
                sample = super().__getitem__(index)
                img_bytes = sample['img']
                label = int(sample['label'])
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                if self._transform:
                    img = self._transform(img)
                return img, label

        return _TransformedStreamingDataset(
            transform=transform,
            local=streaming_dir,
            shuffle=shuffle,
            batch_size=batch_size,
        )

"""AugmentMXDataset v2 — 支持 numpy/tensor 链路的 v2 augmenters.

v2 augmenters 接收 numpy 输入而非 PIL, decode 直接返回 numpy,
省掉 decode→PIL→numpy 的冗余转换.

对于 GPU augmenter, 输入是 tensor (3, H, W) uint8.
"""

import numbers
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .base_dataset import MXFaceDataset
from .recordio_reader import RecordIOReader
from data_augs import make_augmenter


class AugmentMXDatasetV2(MXFaceDataset):
    """V2 dataset: decode→numpy→augment→tensor, 零 PIL 开销."""

    def __init__(self, root_dir, local_rank, augmentation_version='basic_v2_numpy', aug_params=None):
        super().__init__(root_dir, local_rank)
        print(f'AugmentMXDatasetV2: augmentation_version={augmentation_version}')
        self.augmenter = make_augmenter(augmentation_version, aug_params)
        self.is_gpu_augmenter = 'gpu' in augmentation_version

    def __getitem__(self, index):
        if self.is_gpu_augmenter:
            return self._getitem_gpu(index)
        else:
            return self._getitem_numpy(index)

    def _getitem_numpy(self, index):
        """numpy/cv2 augmenters: decode→numpy→augment→tensor(normalized)."""
        sample_np, label = self.read_sample_numpy(index)

        # augment (numpy in → numpy out, or tuple)
        result = self.augmenter.augment(sample_np)
        if isinstance(result, tuple):
            aug_np, theta = result
        else:
            aug_np = result
            theta = None

        # numpy (H, W, 3) uint8 → tensor (3, H, W) float32 normalized
        sample_t = torch.from_numpy(aug_np.transpose(2, 0, 1).copy()).float()
        sample_t.div_(255.0).sub_(0.5).div_(0.5)  # normalize to [-1, 1]

        if theta is not None:
            placeholder = 0
            assert theta.shape == (2, 3)
            return sample_t, placeholder, label, theta
        else:
            return sample_t, label

    def _getitem_gpu(self, index):
        """GPU augmenter: decode→tensor(uint8), augment 在 GPU batch 级别做.

        这里只做 decode + to_tensor, augmentation 留给 training loop.
        返回 uint8 tensor 以节省内存和传输带宽.
        """
        sample_np, label = self.read_sample_numpy(index)

        # numpy (H, W, 3) uint8 → tensor (3, H, W) uint8
        sample_t = torch.from_numpy(sample_np.transpose(2, 0, 1).copy())

        return sample_t, label

    def read_sample_numpy(self, index):
        """Read sample and return numpy array directly (skip PIL)."""
        info_index = self.info.index[index]
        idx = self.imgidx[info_index]
        s = self.imgrec.read_idx(idx)
        header, img_bytes = RecordIOReader.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample_np = RecordIOReader.decode_image_numpy(img_bytes)
        return sample_np, label

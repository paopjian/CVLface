"""BasicAugmenter v2 — torchvision GPU 链路.

设计思路:
- 解码后直接转为 tensor (不经过 PIL)
- augmentation 在 GPU 上批量执行 (利用 GPU 空闲时间)
- 需要配合修改 DataLoader collate: worker 只做 decode→tensor,
  augment 在 GPU 上对整个 batch 执行

使用方式:
  方案A (per-sample, CPU worker 内):
    augmenter.augment(img_tensor)  # img_tensor: (3, H, W) uint8
  方案B (batch-level, GPU, 推荐):
    augmenter.augment_batch(batch_tensor)  # (B, 3, H, W) uint8 on GPU

输入: torch.Tensor (3, H, W) uint8 或 (B, 3, H, W) uint8
输出: torch.Tensor (3, H, W) float32 normalized 或 (B, 3, H, W) float32
"""

import torch
import torch.nn.functional as F
import numpy as np


class BasicAugmenterV2GPU:
    """GPU-accelerated augmenter using pure torch operations.

    All ops use torch tensors — no PIL, no numpy in the hot path.
    Can run on CPU (in DataLoader workers) or GPU (batch-level).
    """

    def __init__(self, crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob):
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

    def augment(self, sample):
        """Per-sample augmentation (CPU or GPU).

        Args:
            sample: torch.Tensor (3, H, W) uint8
        Returns:
            torch.Tensor (3, H, W) uint8
        """
        # crop with zero padding
        if np.random.random() < self.crop_augmentation_prob:
            sample = self.crop_augment(sample)

        # low resolution
        if np.random.random() < self.low_res_augmentation_prob:
            sample = self.low_res_augmentation(sample)

        # photometric
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)

        # random horizontal flip
        if np.random.random() < 0.5:
            sample = sample.flip(-1)

        return sample

    def augment_batch(self, batch):
        """Batch-level augmentation on GPU — 每张图独立随机.

        Args:
            batch: torch.Tensor (B, 3, H, W) uint8 on GPU
        Returns:
            torch.Tensor (B, 3, H, W) float32 normalized [-1, 1]
        """
        B = batch.shape[0]
        results = []
        for i in range(B):
            img = self.augment(batch[i])
            results.append(img)
        out = torch.stack(results, dim=0)
        # normalize: uint8 [0,255] → float32 [-1, 1]
        out = out.float().div_(255.0).sub_(0.5).div_(0.5)
        return out

    def crop_augment(self, img):
        """RandomResizedCrop with zero padding. Input: (3, H, W) uint8."""
        _, h, w = img.shape
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(0.2, 1.0) * area
            aspect_ratio = np.exp(np.random.uniform(np.log(0.75), np.log(4.0 / 3.0)))
            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))
            if 0 < new_w <= w and 0 < new_h <= h:
                i = np.random.randint(0, h - new_h + 1)
                j = np.random.randint(0, w - new_w + 1)
                break
        else:
            new_h = min(h, w)
            new_w = new_h
            i = (h - new_h) // 2
            j = (w - new_w) // 2

        new = torch.zeros_like(img)
        new[:, i:i + new_h, j:j + new_w] = img[:, i:i + new_h, j:j + new_w]
        return new

    def low_res_augmentation(self, img):
        """Resize down then up via torch interpolate. Input: (3, H, W) uint8."""
        _, h, w = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = max(int(side_ratio * h), 1)

        # need float for interpolate
        img_f = img.unsqueeze(0).float()  # (1, 3, H, W)
        small = F.interpolate(img_f, size=(small_side, small_side), mode='bilinear', align_corners=False)
        aug = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
        return aug.squeeze(0).clamp_(0, 255).to(torch.uint8)

    def photometric_augmentation(self, img):
        """ColorJitter in torch. Input: (3, H, W) uint8."""
        ops = [0, 1, 2]
        np.random.shuffle(ops)

        img_f = img.float()  # (3, H, W)
        for op in ops:
            if op == 0:
                # brightness
                factor = np.random.uniform(0.5, 1.5)
                img_f.mul_(factor)
            elif op == 1:
                # contrast: blend with mean
                factor = np.random.uniform(0.5, 1.5)
                mean = img_f.mean()
                img_f.mul_(factor).add_(mean * (1.0 - factor))
            elif op == 2:
                # saturation: blend with grayscale
                factor = np.random.uniform(0.5, 1.5)
                # luma weights for RGB
                gray = img_f[0] * 0.2989 + img_f[1] * 0.5870 + img_f[2] * 0.1141
                img_f.mul_(factor)
                img_f.add_(gray.unsqueeze(0) * (1.0 - factor))

        img_f.clamp_(0, 255)
        return img_f.to(torch.uint8)

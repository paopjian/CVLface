"""BasicAugmenter v2 — 纯 numpy 链路, 零 PIL 依赖.

输入: numpy array (H, W, 3) uint8 (直接来自 TurboJPEG decode)
输出: numpy array (H, W, 3) uint8
后续: torch.from_numpy → permute → float → normalize (在 dataset 层完成)

相比 v1 的优化:
- 去掉所有 PIL Image 中间转换 (省 ~60-80 us/image)
- crop/photometric/hflip 全部用 numpy 原地操作
- low_res 复用 cv2.resize (已经是 numpy)
"""

import numpy as np
import cv2


class BasicAugmenterV2Numpy:

    def __init__(self, crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob):
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

    def augment(self, sample):
        """
        Args:
            sample: numpy array (H, W, 3) uint8, RGB
        Returns:
            numpy array (H, W, 3) uint8, RGB
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
            sample = sample[:, ::-1, :].copy()

        return sample

    def crop_augment(self, img):
        """RandomResizedCrop: 随机裁剪一块区域, 放到零填充的画布上."""
        h, w = img.shape[:2]
        # sample crop parameters (same logic as torchvision RandomResizedCrop)
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
            # fallback: center crop
            new_h = min(h, w)
            new_w = new_h
            i = (h - new_h) // 2
            j = (w - new_w) // 2

        # zero-padded canvas with crop placed at original position
        new = np.zeros_like(img)
        new[i:i + new_h, j:j + new_w, :] = img[i:i + new_h, j:j + new_w, :]
        return new

    def low_res_augmentation(self, img):
        """Resize down then up to simulate low resolution."""
        h, w = img.shape[:2]
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = max(int(side_ratio * h), 1)
        interpolation_down = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        interpolation_up = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation_down)
        aug_img = cv2.resize(small_img, (w, h), interpolation=interpolation_up)
        return aug_img

    def photometric_augmentation(self, img):
        """ColorJitter: brightness, contrast, saturation — 纯 numpy 实现."""
        # random order
        ops = [0, 1, 2]
        np.random.shuffle(ops)

        img = img.astype(np.float32)
        for op in ops:
            if op == 0:
                # brightness: multiply by factor in [0.5, 1.5]
                factor = np.random.uniform(0.5, 1.5)
                img = img * factor
            elif op == 1:
                # contrast: blend with mean gray
                factor = np.random.uniform(0.5, 1.5)
                mean = img.mean()
                img = img * factor + mean * (1.0 - factor)
            elif op == 2:
                # saturation: blend with grayscale
                factor = np.random.uniform(0.5, 1.5)
                gray = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1141
                gray = gray[:, :, np.newaxis]
                img = img * factor + gray * (1.0 - factor)

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

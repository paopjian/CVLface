"""BasicAugmenter v2 — 全 cv2 链路.

输入: numpy array (H, W, 3) uint8 RGB (来自 TurboJPEG decode)
输出: numpy array (H, W, 3) uint8 RGB
后续: torch.from_numpy → permute → float → normalize

相比 v1 的优化:
- 零 PIL 依赖, 全程 numpy/cv2
- cv2 的 SIMD 优化比 PIL 快 (特别是 resize, color conversion)
- 无格式转换开销
"""

import numpy as np
import cv2


class BasicAugmenterV2CV2:

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
        if np.random.random() < self.crop_augmentation_prob:
            sample = self.crop_augment(sample)

        if np.random.random() < self.low_res_augmentation_prob:
            sample = self.low_res_augmentation(sample)

        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)

        if np.random.random() < 0.5:
            sample = cv2.flip(sample, 1)  # horizontal flip, faster than numpy slicing

        return sample

    def crop_augment(self, img):
        """RandomResizedCrop with zero padding."""
        h, w = img.shape[:2]
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

        new = np.zeros_like(img)
        new[i:i + new_h, j:j + new_w, :] = img[i:i + new_h, j:j + new_w, :]
        return new

    def low_res_augmentation(self, img):
        """Resize down then up."""
        h, w = img.shape[:2]
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = max(int(side_ratio * h), 1)
        interp_down = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        interp_up = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interp_down)
        return cv2.resize(small_img, (w, h), interpolation=interp_up)

    def photometric_augmentation(self, img):
        """ColorJitter via HSV color space (cv2 optimized)."""
        # Convert RGB → HSV for saturation/brightness, keep in float32
        # cv2 expects BGR for cvtColor, but we work in RGB
        # Use manual approach to avoid RGB→BGR→HSV→BGR→RGB round-trips

        ops = [0, 1, 2]
        np.random.shuffle(ops)

        img = img.astype(np.float32)
        for op in ops:
            if op == 0:
                # brightness
                factor = np.random.uniform(0.5, 1.5)
                img *= factor
            elif op == 1:
                # contrast: blend with channel-wise mean
                factor = np.random.uniform(0.5, 1.5)
                mean = img.mean(axis=(0, 1), keepdims=True)
                img = img * factor + mean * (1.0 - factor)
            elif op == 2:
                # saturation: blend with luminance
                factor = np.random.uniform(0.5, 1.5)
                # ITU-R BT.601 luma
                gray = np.dot(img, [0.2989, 0.5870, 0.1141])
                gray = gray[:, :, np.newaxis]
                img = img * factor + gray * (1.0 - factor)

        np.clip(img, 0, 255, out=img)
        return img.astype(np.uint8)

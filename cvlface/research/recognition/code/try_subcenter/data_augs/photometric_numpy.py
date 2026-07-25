"""PhotometricRandAugment numpy 实现 — gridsample_augmenter_v2_numpy 的续文件.

由于文件长度限制, PhotometricRandAugmentNumpy 单独存放在此文件.
gridsample_augmenter_v2_numpy.py 从这里 import.
"""

import numpy as np
import cv2


class PhotometricRandAugmentNumpy:
    """PhotometricRandAugment — 纯 numpy/cv2, 不依赖 torchvision.transforms.functional.

    实现和 v1 PhotometricRandAugment 相同的 ops:
    Identity, Brightness, Saturate, Contrast, Sharpness, Equalize, Grayscale
    """

    def __init__(self, num_ops=2, magnitude=9, magnitude_offset=4, num_magnitude_bins=31):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.magnitude_offset = magnitude_offset
        self.num_magnitude_bins = num_magnitude_bins
        self.op_names = ['Identity', 'Brightness', 'Saturate', 'Contrast', 'Sharpness', 'Equalize', 'Grayscale']
        # pre-compute magnitude linspace
        self._mag_linspace = np.linspace(0.0, 0.9, num_magnitude_bins)

    def _sample_magnitude(self, signed):
        idx = np.random.randint(
            max(self.magnitude - self.magnitude_offset, 0),
            min(self.magnitude + self.magnitude_offset, self.num_magnitude_bins - 1) + 1)
        mag = float(self._mag_linspace[idx])
        if signed and np.random.random() < 0.5:
            mag *= -1.0
        return mag

    def augment(self, img, param=None):
        """Input/output: numpy (H, W, 3) uint8 RGB."""
        if param is None:
            param = self._sample_param()
        for op_name, magnitude in param:
            img = self._apply_op(img, op_name, magnitude)
        return img

    def _sample_param(self):
        ops = []
        for _ in range(self.num_ops):
            op_name = np.random.choice(self.op_names)
            # reduce probability of Equalize/Grayscale (same as v1)
            if op_name in ['Equalize', 'Grayscale']:
                op_name = np.random.choice(self.op_names)
                if op_name in ['Equalize', 'Grayscale']:
                    op_name = np.random.choice(self.op_names)

            signed = op_name in ['Brightness', 'Saturate', 'Contrast', 'Sharpness']
            if op_name in ['Identity', 'Equalize', 'Grayscale']:
                mag = 0.0
            else:
                mag = self._sample_magnitude(signed)
            ops.append((op_name, mag))
        return ops

    def _apply_op(self, img, op_name, magnitude):
        if op_name == 'Identity':
            return img
        elif op_name == 'Brightness':
            # Same as F.adjust_brightness: multiply all channels
            img_f = img.astype(np.float32)
            img_f *= (1.0 + magnitude)
            np.clip(img_f, 0, 255, out=img_f)
            return img_f.astype(np.uint8)
        elif op_name == 'Saturate':
            # Blend with grayscale
            img_f = img.astype(np.float32)
            gray = img_f[:, :, 0] * 0.2989 + img_f[:, :, 1] * 0.5870 + img_f[:, :, 2] * 0.1141
            factor = 1.0 + magnitude
            img_f = img_f * factor + gray[:, :, np.newaxis] * (1.0 - factor)
            np.clip(img_f, 0, 255, out=img_f)
            return img_f.astype(np.uint8)
        elif op_name == 'Contrast':
            # Blend with mean gray image
            img_f = img.astype(np.float32)
            mean = img_f.mean()
            factor = 1.0 + magnitude
            img_f = img_f * factor + mean * (1.0 - factor)
            np.clip(img_f, 0, 255, out=img_f)
            return img_f.astype(np.uint8)
        elif op_name == 'Sharpness':
            # Unsharp mask: blend original with blurred
            factor = 1.0 + magnitude
            # kernel for smoothing (like PIL ImageFilter.SMOOTH)
            kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.0
            smooth = cv2.filter2D(img, -1, kernel)
            img_f = img.astype(np.float32)
            smooth_f = smooth.astype(np.float32)
            result = img_f * factor + smooth_f * (1.0 - factor)
            np.clip(result, 0, 255, out=result)
            return result.astype(np.uint8)
        elif op_name == 'Equalize':
            # Histogram equalization per channel
            result = np.empty_like(img)
            for c in range(3):
                result[:, :, c] = cv2.equalizeHist(img[:, :, c])
            return result
        elif op_name == 'Grayscale':
            # Convert to gray and replicate 3 channels
            gray = (img[:, :, 0].astype(np.float32) * 0.2989 +
                    img[:, :, 1].astype(np.float32) * 0.5870 +
                    img[:, :, 2].astype(np.float32) * 0.1141)
            gray = gray.astype(np.uint8)
            return np.stack([gray, gray, gray], axis=2)
        return img

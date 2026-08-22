"""GridSampleAugmenter v2 — 纯 numpy/cv2 链路, 零 PIL 依赖.

输入: numpy array (H, W, 3) uint8 RGB (来自 TurboJPEG decode)
输出: (numpy array (H, W, 3) uint8, theta tensor (2, 3))

相比 v1 的优化:
- augment_cv2_deterministic 不再包装成 PIL (省一次 Image.fromarray)
- CutoutAugment 全 numpy 实现 (去掉 F.crop + Image.fromarray)
- BlurAugmenter 全 cv2 实现 (去掉 imgaug 依赖, 去掉 np.array(PIL) 转换)
- PhotometricRandAugment 全 numpy 实现 (去掉 F.adjust_* PIL ops)
- 整条链路: numpy in → numpy out, 零 PIL 对象创建
"""

import numpy as np
import cv2
from data_augs.aug_utils import transform_torch
from data_augs.aug_utils import transform_cv2
from data_augs.photometric_numpy import PhotometricRandAugmentNumpy


class GridSampleAugmenterV2Numpy:

    def __init__(self, aug_params, input_size=112):
        print('GridSampleAugmenterV2Numpy')
        self.aug_params = aug_params
        self.input_size = input_size
        self.photo_aug = PhotometricRandAugmentNumpy(
            num_ops=self.aug_params['photometric_num_ops'],
            magnitude=self.aug_params['photometric_magnitude'],
            magnitude_offset=self.aug_params['photometric_magnitude_offset'],
            num_magnitude_bins=self.aug_params['photometric_num_magnitude_bins'])
        self.blur_aug = BlurAugmenterNumpy(
            magnitude=self.aug_params['blur_magnitude'],
            prob=self.aug_params['blur_prob'])
        self.cutout = CutoutAugmentNumpy(aug_params['cutout_prob'])

    def augment(self, sample):
        """
        Args:
            sample: numpy array (H, W, 3) uint8, RGB
        Returns:
            (numpy array (H, W, 3) uint8, theta tensor (2, 3))
        """
        image_np = sample  # already numpy

        # affine transform
        params = transform_torch.sample_param(
            scale_min=self.aug_params['scale_min'],
            scale_max=self.aug_params['scale_max'],
            rot_prob=self.aug_params['rot_prob'],
            max_rot=self.aug_params['max_rot'],
            hflip_prob=self.aug_params['hflip_prob'],
            extra_offset=self.aug_params['extra_offset'],
        )
        mat = transform_cv2.generate_transform_cv2(
            image_np, self.input_size, self.input_size, **params)
        # cv2.warpPerspective directly returns numpy — no PIL wrap
        aug_sample = cv2.warpPerspective(
            image_np, mat, (self.input_size, self.input_size), borderValue=0)

        # corresponding theta for grid_sample
        align_input_theta = transform_torch.generate_transform_torch(
            image_np, self.input_size, self.input_size, **params)
        align_input_theta = align_input_theta.squeeze(0)

        # cutout (numpy in/out)
        aug_sample = self.cutout.augment(aug_sample)

        # blur (numpy in/out)
        aug_sample = self.blur_aug.augment(aug_sample)

        # photometric (numpy in/out)
        aug_sample = self.photo_aug.augment(aug_sample)

        return aug_sample, align_input_theta


class CutoutAugmentNumpy:
    """Cutout augmentation — 纯 numpy 实现."""

    def __init__(self, cutout_prob):
        self.cutout_prob = cutout_prob

    def augment(self, img):
        """Input/output: numpy (H, W, 3) uint8."""
        if np.random.random() >= self.cutout_prob:
            return img

        h, w = img.shape[:2]
        if np.random.random() < 0.05:
            # coarse dropout: random rectangular holes
            result = img.copy()
            num_holes = np.random.randint(12, 21)
            for _ in range(num_holes):
                hole_h = np.random.randint(1, 17)
                hole_w = np.random.randint(1, 17)
                y = np.random.randint(0, h - hole_h + 1)
                x = np.random.randint(0, w - hole_w + 1)
                result[y:y + hole_h, x:x + hole_w, :] = 0
            return result
        else:
            # random resized crop onto zero canvas
            area = h * w
            for _ in range(10):
                target_area = np.random.uniform(0.2, 1.0) * area
                aspect_ratio = np.exp(np.random.uniform(
                    np.log(0.75), np.log(4.0 / 3.0)))
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


class BlurAugmenterNumpy:
    """Blur augmentation — 纯 cv2 实现, 不依赖 imgaug."""

    def __init__(self, magnitude=0.5, prob=0.2):
        self.magnitude = magnitude
        self.prob = prob

    def augment(self, img, param=None):
        """Input/output: numpy (H, W, 3) uint8."""
        if param is None:
            if np.random.random() >= self.prob:
                return img
            param = self._sample_param()

        method = param[0]
        if method == 'skip':
            return img
        elif method == 'avg':
            k = param[1]
            k = max(k, 1)
            if k % 2 == 0:
                k += 1
            return cv2.blur(img, (k, k))
        elif method == 'gaussian':
            sigma = param[1]
            # kernel size must be odd
            k = int(np.ceil(sigma * 3)) * 2 + 1
            k = max(k, 3)
            return cv2.GaussianBlur(img, (k, k), sigma)
        elif method == 'resize':
            side_ratio = param[1]
            interp = param[2]
            h, w = img.shape[:2]
            small_side = max(int(side_ratio * h), 1)
            small = cv2.resize(img, (small_side, small_side), interpolation=interp[0])
            return cv2.resize(small, (w, h), interpolation=interp[1])
        return img

    def _sample_param(self):
        blur_method = np.random.choice(
            ['avg', 'gaussian', 'resize', 'resize', 'resize', 'resize',
             'resize', 'resize', 'resize', 'resize'])
        if blur_method == 'avg':
            k = np.random.randint(1, int(10 * self.magnitude))
            return [blur_method, k]
        elif blur_method == 'gaussian':
            sigma = np.random.random() * 4 * self.magnitude
            return [blur_method, sigma]
        elif blur_method == 'resize':
            side_ratio = np.random.uniform(1.0 - 0.8 * self.magnitude, 1.0)
            interp1 = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                        cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
            interp2 = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                        cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
            return [blur_method, side_ratio, [interp1, interp2]]
        return ['skip']

"""GridSampleAugmenter v2 — torchvision GPU 链路.

设计: 利用 GPU 空闲周期做 augmentation, 减轻 CPU worker 负担.

方案:
- DataLoader worker 只做 decode + numpy→tensor (零 augmentation)
- GPU 端对整 batch 执行 warp + photometric + cutout + blur
- 用 torch.nn.functional.grid_sample 替代 cv2.warpPerspective
- photometric 用 torch in-place 操作

输入: torch.Tensor (3, H, W) uint8
输出: (torch.Tensor (3, H, W) uint8, theta tensor (2, 3))

注意: 此 augmenter 的 augment() 方法在 CPU 上也能跑 (per-sample),
      但推荐用 augment_batch() 在 GPU 上批量处理.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GridSampleAugmenterV2GPU:

    def __init__(self, aug_params, input_size=112):
        print('GridSampleAugmenterV2GPU')
        self.aug_params = aug_params
        self.input_size = input_size

    def augment(self, sample):
        """Per-sample augmentation (CPU tensor).

        Args:
            sample: torch.Tensor (3, H, W) uint8
        Returns:
            (torch.Tensor (3, H, W) uint8, theta tensor (2, 3))
        """
        c, h, w = sample.shape

        # sample transform params
        params = self._sample_affine_params()
        theta = self._build_theta(h, w, params)  # (1, 2, 3)

        # apply affine via grid_sample
        img_f = sample.unsqueeze(0).float()  # (1, 3, H, W)
        grid = F.affine_grid(theta, [1, c, self.input_size, self.input_size], align_corners=True)
        warped = F.grid_sample(img_f, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        aug_sample = warped.squeeze(0).clamp_(0, 255).to(torch.uint8)  # (3, H, W)

        # cutout
        if np.random.random() < self.aug_params.get('cutout_prob', 0.2):
            aug_sample = self._cutout(aug_sample)

        # blur (simple box blur via avg_pool2d)
        if np.random.random() < self.aug_params.get('blur_prob', 0.2):
            aug_sample = self._blur(aug_sample)

        # photometric
        aug_sample = self._photometric(aug_sample)

        return aug_sample, theta.squeeze(0)

    def augment_batch(self, batch, device=None):
        """Batch-level GPU augmentation.

        Args:
            batch: torch.Tensor (B, 3, H, W) uint8, on GPU
        Returns:
            (torch.Tensor (B, 3, H, W) float32 normalized, thetas (B, 2, 3))
        """
        B, c, h, w = batch.shape
        if device is None:
            device = batch.device

        thetas = []
        results = []
        for i in range(B):
            aug_img, theta = self.augment(batch[i])
            results.append(aug_img)
            thetas.append(theta)

        out = torch.stack(results, dim=0).float()
        out.div_(255.0).sub_(0.5).div_(0.5)  # normalize to [-1, 1]
        thetas = torch.stack(thetas, dim=0)
        return out, thetas

    def _sample_affine_params(self):
        scale = np.random.uniform(
            self.aug_params.get('scale_min', 1.0),
            self.aug_params.get('scale_max', 1.0))
        if np.random.random() < self.aug_params.get('rot_prob', 0.0):
            angle = np.random.uniform(
                -self.aug_params.get('max_rot', 0),
                self.aug_params.get('max_rot', 0))
        else:
            angle = 0.0
        hflip = np.random.random() < self.aug_params.get('hflip_prob', 0.5)
        extra_offset = np.random.uniform(0, self.aug_params.get('extra_offset', 0.0))
        tx = np.random.uniform(-extra_offset, extra_offset)
        ty = np.random.uniform(-extra_offset, extra_offset)
        return {'scale': scale, 'angle': angle, 'hflip': hflip, 'tx': tx, 'ty': ty}

    def _build_theta(self, h, w, params):
        """Build 2x3 affine theta for grid_sample."""
        angle_rad = np.deg2rad(params['angle'])
        s = params['scale']
        cos_a = np.cos(angle_rad) * s
        sin_a = np.sin(angle_rad) * s

        # rotation + scale
        theta = np.array([
            [cos_a, -sin_a, params['tx']],
            [sin_a, cos_a, params['ty']]
        ], dtype=np.float32)

        # horizontal flip
        if params['hflip']:
            theta[0, 0] *= -1
            theta[0, 1] *= -1

        return torch.from_numpy(theta).unsqueeze(0)  # (1, 2, 3)

    def _cutout(self, img):
        """Random rectangular cutout. Input: (3, H, W) uint8."""
        _, h, w = img.shape
        # random resized crop onto zero canvas
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(0.2, 1.0) * area
            ar = np.exp(np.random.uniform(np.log(0.75), np.log(4.0 / 3.0)))
            new_w = int(round(np.sqrt(target_area * ar)))
            new_h = int(round(np.sqrt(target_area / ar)))
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

    def _blur(self, img):
        """Simple resize-based blur. Input: (3, H, W) uint8."""
        _, h, w = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = max(int(side_ratio * h), 1)
        img_f = img.unsqueeze(0).float()
        small = F.interpolate(img_f, size=(small_side, small_side), mode='bilinear', align_corners=False)
        aug = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
        return aug.squeeze(0).clamp_(0, 255).to(torch.uint8)

    def _photometric(self, img):
        """Random photometric augmentation. Input: (3, H, W) uint8."""
        num_ops = self.aug_params.get('photometric_num_ops', 2)
        magnitude = self.aug_params.get('photometric_magnitude', 9)
        mag_offset = self.aug_params.get('photometric_magnitude_offset', 4)
        num_bins = self.aug_params.get('photometric_num_magnitude_bins', 31)
        mag_linspace = np.linspace(0.0, 0.9, num_bins)

        img_f = img.float()
        ops = ['Brightness', 'Saturate', 'Contrast', 'Sharpness', 'Identity', 'Equalize', 'Grayscale']

        for _ in range(num_ops):
            op = np.random.choice(ops)
            # reduce Equalize/Grayscale probability
            if op in ['Equalize', 'Grayscale']:
                op = np.random.choice(ops)
                if op in ['Equalize', 'Grayscale']:
                    op = np.random.choice(ops)

            idx = np.random.randint(max(magnitude - mag_offset, 0),
                                    min(magnitude + mag_offset, num_bins - 1) + 1)
            mag = float(mag_linspace[idx])
            if np.random.random() < 0.5:
                mag *= -1.0

            if op == 'Brightness':
                img_f.mul_(1.0 + mag)
            elif op == 'Contrast':
                mean = img_f.mean()
                img_f.mul_(1.0 + mag).add_(mean * (-mag))
            elif op == 'Saturate':
                gray = img_f[0] * 0.2989 + img_f[1] * 0.5870 + img_f[2] * 0.1141
                factor = 1.0 + mag
                img_f.mul_(factor)
                img_f.add_(gray.unsqueeze(0) * (1.0 - factor))
            elif op == 'Grayscale':
                gray = img_f[0] * 0.2989 + img_f[1] * 0.5870 + img_f[2] * 0.1141
                img_f[0] = gray
                img_f[1] = gray
                img_f[2] = gray
            # Identity/Sharpness/Equalize: skip for GPU simplicity

        img_f.clamp_(0, 255)
        return img_f.to(torch.uint8)

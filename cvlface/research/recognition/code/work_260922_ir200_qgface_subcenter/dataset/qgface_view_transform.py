import cv2
import numpy as np
import torch


class QGFaceViewTransform:
    """Build one QGFace low-quality view with a numpy/OpenCV-only path."""

    _INTERPOLATIONS = (
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )

    def __init__(
        self,
        output_size=112,
        low_res_scale=(0.1, 0.5),
        jpeg_quality=75,
        crop_probability=0.2,
        photometric_probability=0.2,
        rotation_probability=0.0,
    ):
        self.output_size = output_size
        self.low_res_scale = low_res_scale
        self.jpeg_quality = jpeg_quality
        self.crop_probability = crop_probability
        self.photometric_probability = photometric_probability
        self.rotation_probability = rotation_probability

    def _low_resolution(self, image):
        ratio = np.random.uniform(self.low_res_scale[0], self.low_res_scale[1])
        low_resolution_size = max(1, int(self.output_size * ratio))
        image = cv2.resize(
            image,
            (low_resolution_size, low_resolution_size),
            interpolation=np.random.choice(self._INTERPOLATIONS),
        )
        if self.jpeg_quality is not None:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(
                '.jpg',
                bgr_image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not success:
                raise RuntimeError('Failed to encode QGFace JPEG augmentation')
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(
            image,
            (self.output_size, self.output_size),
            interpolation=np.random.choice(self._INTERPOLATIONS),
        )

    def _random_resized_crop(self, image):
        height, width = image.shape[:2]
        area = height * width
        for _ in range(10):
            target_area = np.random.uniform(0.8, 1.0) * area
            aspect_ratio = np.exp(np.random.uniform(np.log(0.75), np.log(1.33)))
            crop_width = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(np.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = np.random.randint(0, height - crop_height + 1)
                left = np.random.randint(0, width - crop_width + 1)
                crop = image[top : top + crop_height, left : left + crop_width]
                return cv2.resize(
                    crop,
                    (self.output_size, self.output_size),
                    interpolation=cv2.INTER_LINEAR,
                )
        return image

    @staticmethod
    def _photometric(image):
        operations = np.random.permutation(3)
        image = image.astype(np.float32)
        for operation in operations:
            factor = np.random.uniform(0.8, 1.2)
            if operation == 0:
                image *= factor
            elif operation == 1:
                grayscale_mean = (
                    image[:, :, 0] * 0.2989
                    + image[:, :, 1] * 0.5870
                    + image[:, :, 2] * 0.1141
                ).mean()
                image = image * factor + grayscale_mean * (1 - factor)
            else:
                grayscale = (
                    image[:, :, 0] * 0.2989
                    + image[:, :, 1] * 0.5870
                    + image[:, :, 2] * 0.1141
                )[:, :, None]
                image = image * factor + grayscale * (1 - factor)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _rotate(self, image):
        angle = np.random.uniform(-45, 45)
        center = ((self.output_size - 1) / 2, (self.output_size - 1) / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def __call__(self, normalized_image):
        image = normalized_image.detach().cpu()
        if image.dtype == torch.uint8:
            image = image.permute(1, 2, 0).numpy()
        else:
            image = image.float().mul(0.5).add(0.5).clamp(0, 1)
            image = image.mul(255).round().byte().permute(1, 2, 0).numpy()

        image = self._low_resolution(image)
        if np.random.random() < self.crop_probability:
            image = self._random_resized_crop(image)
        if np.random.random() < self.photometric_probability:
            image = self._photometric(image)
        if np.random.random() < self.rotation_probability:
            image = self._rotate(image)
        if np.random.random() < 0.5:
            image = image[:, ::-1].copy()

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return tensor.div_(255.0).sub_(0.5).div_(0.5)

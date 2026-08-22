import torch
from torch.utils.data import Dataset


class ContrastiveViewDataset(Dataset):
    """Return the QGFace low-quality/original pair for one identity."""

    def __init__(self, dataset, view_transform=None):
        self.dataset = dataset
        self.view_transform = view_transform
        self.color_space = dataset.color_space
        self.contrastive_views = True

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _image_and_label(sample):
        if len(sample) == 2:
            return sample[0], sample[1]
        if len(sample) == 4:
            return sample[0], sample[2]
        if len(sample) == 7:
            return sample[0], sample[1]
        raise ValueError(f"Unsupported sample format with {len(sample)} fields")

    def __getitem__(self, index):
        image, query_label = self._image_and_label(self.dataset[index])
        if self.view_transform is None:
            query_image = image
        else:
            query_image = self.view_transform(image)

        key_image = image.clone()
        if torch.rand(()) < 0.5:
            key_image = key_image.flip(-1)
        return query_image, key_image, query_label

    def set_augmentation(self, enabled):
        if hasattr(self.dataset, "set_augmentation"):
            self.dataset.set_augmentation(enabled)

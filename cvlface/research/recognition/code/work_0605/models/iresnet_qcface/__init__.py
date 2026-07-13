"""CVLface model wrapper for the original-QCFace IResNet backbone.

Loads weights from the original QCFace checkpoint format, whose state_dict has
keys ``backbone.<layer>...`` (backbone) and ``head.weight`` (the QCFace margin
head). Only the backbone is kept; keys are remapped ``backbone.X -> net.X``.

IMPORTANT — normalisation: the QCFace pretrained model was trained with the
custom mean/std below (see QCFace repo validation dataloaders), NOT the
[0.5,0.5,0.5] used by CVLface's own iresnet. Using the wrong normalisation
makes the pretrained weights produce garbage embeddings.
"""

import os
import torch
from torch import nn
from torchvision import transforms

from ..base import BaseModel
from ..base.utils import load_state_dict_from_path
from .model import iresnet18, iresnet50, iresnet100

# QCFace training/eval normalisation (from QCFace repo dataloaders).
QCFACE_MEAN = [0.5312, 0.4265, 0.3753]
QCFACE_STD = [0.2873, 0.2555, 0.2496]


class IResNetQCFaceModel(BaseModel):

    def __init__(self, net, config):
        super(IResNetQCFaceModel, self).__init__(config)
        self.net = net
        self.config = config

    @classmethod
    def from_config(cls, config):
        if config.name == 'ir18_qcface':
            net = iresnet18(num_classes=config.output_dim)
        elif config.name == 'ir50_qcface':
            net = iresnet50(num_classes=config.output_dim)
        elif config.name == 'ir101_qcface':
            net = iresnet100(num_classes=config.output_dim)
        else:
            raise NotImplementedError(f"unknown qcface backbone: {config.name}")
        model = cls(net, config)
        model.eval()
        return model

    def forward(self, x):
        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x)

    def make_train_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=QCFACE_MEAN, std=QCFACE_STD),
        ])

    def make_test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=QCFACE_MEAN, std=QCFACE_STD),
        ])

    def load_state_dict_from_path(self, pretrained_model_path):
        """Load either an original QCFace ckpt (backbone.*/head.*) or a CVLface
        model.pt saved by this framework (net.*)."""
        state_dict = load_state_dict_from_path(pretrained_model_path)
        # Original QCFace checkpoint wraps params under 'state_dict'.
        if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']

        # Remap backbone.* -> net.* and drop the classification head.
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                remapped['net.' + k[len('backbone.'):]] = v
            elif k.startswith('head.'):
                continue  # margin head not part of the backbone
            else:
                remapped[k] = v  # already net.* (CVLface-saved ckpt)

        self_keys = set(self.state_dict().keys())
        matched = len(self_keys.intersection(remapped.keys()))
        print('compatible keys in state_dict', matched, '/', len(remapped))
        print('Check\n\n')
        result = self.load_state_dict(remapped, strict=False)
        print(result)
        print(f"Loaded QCFace pretrained model from {pretrained_model_path}")


def load_model(model_config):
    return IResNetQCFaceModel.from_config(model_config)

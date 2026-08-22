import os
from typing import Union

import torch

from ..base import BaseClassifier, load_state_dict_from_path
from .fc import FC


class FCClassifier(BaseClassifier):

    def __init__(self, classifier, config, rank, world_size):
        super(FCClassifier, self).__init__()
        self.classifier = classifier
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = True

    @classmethod
    def from_config(cls, classifier_cfg, margin_loss_fn, model_cfg, num_classes, rank, world_size):
        if classifier_cfg.name == 'fc':
            classifier = FC(
                margin_loss=margin_loss_fn,
                embedding_size=model_cfg.output_dim,
                num_classes=num_classes,
            )
        else:
            raise NotImplementedError

        model = cls(classifier, classifier_cfg, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        loss = self.classifier(local_embeddings, local_labels)
        return loss

    @torch.no_grad()
    def route_subcenters(self, labels, embeddings):
        return torch.zeros_like(labels, dtype=torch.long)

    @torch.no_grad()
    def get_class_proxies(self, labels, subcenter_ids=None):
        return self.classifier.weight[labels.long()]

    @torch.no_grad()
    def get_margin_scaler(self, norms):
        margin_loss = self.classifier.margin_softmax
        scaler = (norms.detach().float() - self.classifier.batch_mean.float()) / (
            self.classifier.batch_std.float() + margin_loss.eps
        )
        return (scaler * margin_loss.h).clamp(-1, 1)

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        name: str = 'classifier.pt',
        rank: int = 0,
    ):
        if rank == 0:
            super().save_pretrained(save_dir, name, rank)

    def load_state_dict_from_path(self, pretrained_model_path):
        if os.path.isdir(pretrained_model_path):
            pretrained_model_path = os.path.join(
                pretrained_model_path, 'classifier_rank0.pt'
            )
        elif '_rank' not in os.path.basename(pretrained_model_path):
            save_dir = os.path.dirname(pretrained_model_path)
            save_name = os.path.basename(pretrained_model_path)
            rank_added_name = (
                os.path.splitext(save_name)[0]
                + '_rank0'
                + os.path.splitext(save_name)[1]
            )
            pretrained_model_path = os.path.join(save_dir, rank_added_name)

        state_dict = load_state_dict_from_path(pretrained_model_path)
        result = self.load_state_dict(state_dict, strict=False)
        print('classifier loading result', result)

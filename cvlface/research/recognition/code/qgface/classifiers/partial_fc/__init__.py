import torch
from torch import distributed

from ..base import BaseClassifier
from .partial_fc import PartialFC_V2


class PartialFCClassifier(BaseClassifier):

    def __init__(self, classifier, config, rank, world_size):
        super(PartialFCClassifier, self).__init__()
        self.partial_fc = classifier
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = False

    @classmethod
    def from_config(cls, classifier_cfg, margin_loss_fn, model_cfg, num_classes, rank, world_size):
        if classifier_cfg.name == 'partial_fc':
            classifier = PartialFC_V2(
                rank=rank,
                world_size=world_size,
                margin_loss=margin_loss_fn,
                embedding_size=model_cfg.output_dim,
                num_classes=num_classes,
                sample_rate=classifier_cfg.sample_rate,
                pad_classes=getattr(classifier_cfg, 'pad_classes', True),
            )
        else:
            raise NotImplementedError

        model = cls(classifier, classifier_cfg, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        loss = self.partial_fc(local_embeddings, local_labels)
        return loss

    @torch.no_grad()
    def get_class_proxies(self, labels):
        labels = labels.long()
        proxies = torch.zeros(
            labels.shape[0],
            self.partial_fc.embedding_size,
            device=self.partial_fc.weight.device,
            dtype=self.partial_fc.weight.dtype,
        )
        local_indices = labels - self.partial_fc.class_start
        is_local = (local_indices >= 0) & (local_indices < self.partial_fc.num_local)
        proxies[is_local] = self.partial_fc.weight[local_indices[is_local]]
        if self.world_size > 1:
            distributed.all_reduce(proxies, op=distributed.ReduceOp.SUM)
        return proxies

    @torch.no_grad()
    def get_margin_scaler(self, norms):
        margin_loss = self.partial_fc.margin_softmax
        scaler = (norms.detach().float() - self.partial_fc.batch_mean.float()) / (
            self.partial_fc.batch_std.float() + margin_loss.eps
        )
        return (scaler * margin_loss.h).clamp(-1, 1)


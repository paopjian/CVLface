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
                num_subcenters=getattr(classifier_cfg, 'num_subcenters', 1),
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
    def route_subcenters(self, labels, embeddings):
        labels = labels.long()
        local_indices = labels - self.partial_fc.class_start
        is_local = (local_indices >= 0) & (local_indices < self.partial_fc.num_local)
        subcenter_ids = torch.full_like(labels, -1)
        if is_local.any():
            offsets = torch.arange(
                self.partial_fc.num_subcenters,
                device=labels.device,
            )
            weight_indices = (
                local_indices[is_local, None] * self.partial_fc.num_subcenters
                + offsets[None, :]
            )
            centers = torch.nn.functional.normalize(
                self.partial_fc.weight[weight_indices], dim=2
            )
            features = torch.nn.functional.normalize(embeddings[is_local], dim=1)
            similarities = torch.einsum("bd,bkd->bk", features, centers)
            subcenter_ids[is_local] = similarities.argmax(dim=1)
        if self.world_size > 1:
            distributed.all_reduce(subcenter_ids, op=distributed.ReduceOp.MAX)
        if (subcenter_ids < 0).any():
            raise ValueError("Cannot route labels outside the classifier class range")
        return subcenter_ids

    @torch.no_grad()
    def get_class_proxies(self, labels, subcenter_ids=None):
        labels = labels.long()
        if subcenter_ids is None:
            subcenter_ids = torch.zeros_like(labels)
        subcenter_ids = subcenter_ids.long()
        if ((subcenter_ids < 0) | (
            subcenter_ids >= self.partial_fc.num_subcenters
        )).any():
            raise ValueError("subcenter_ids are outside the configured range")
        proxies = torch.zeros(
            labels.shape[0],
            self.partial_fc.embedding_size,
            device=self.partial_fc.weight.device,
            dtype=self.partial_fc.weight.dtype,
        )
        local_indices = labels - self.partial_fc.class_start
        is_local = (local_indices >= 0) & (local_indices < self.partial_fc.num_local)
        weight_indices = (
            local_indices[is_local] * self.partial_fc.num_subcenters
            + subcenter_ids[is_local]
        )
        proxies[is_local] = self.partial_fc.weight[weight_indices]
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

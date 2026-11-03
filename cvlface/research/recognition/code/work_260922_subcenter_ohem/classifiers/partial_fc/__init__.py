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
                num_subcenters=getattr(classifier_cfg, 'num_subcenters', 1),
                ohem_ratio=getattr(classifier_cfg, 'ohem_ratio', 1.0),
                ohem_min_kept=getattr(classifier_cfg, 'ohem_min_kept', 0),
                ohem_warmup_steps=getattr(classifier_cfg, 'ohem_warmup_steps', 0),
            )
        else:
            raise NotImplementedError

        model = cls(classifier, classifier_cfg, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        loss = self.partial_fc(local_embeddings, local_labels)
        return loss





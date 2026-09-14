from ..base import BaseClassifier
from .partial_fc import PartialFC_V2
from losses.topofr_loss import TopoFRLoss


class PartialFCClassifier(BaseClassifier):

    def __init__(self, classifier, config, rank, world_size):
        super(PartialFCClassifier, self).__init__()
        self.partial_fc = classifier
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = False

        # margin_softmax lives on the inner PartialFC_V2; expose the flag on the
        # wrapper so the training pipeline can read it without reaching inside.
        self.is_topofr = isinstance(getattr(classifier, 'margin_softmax', None), TopoFRLoss)

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
            )
        else:
            raise NotImplementedError

        model = cls(classifier, classifier_cfg, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels, input_images=None, embeddings=None):
        """
        Forward pass with optional TopoFR support

        Args:
            local_embeddings: Feature embeddings from backbone
            local_labels: Ground truth labels
            input_images: Original input images (for TopoFR topological loss)
            embeddings: Same as local_embeddings (for TopoFR, kept for compatibility)
        """
        if self.is_topofr and input_images is not None:
            # Pass additional info for TopoFR
            loss = self.partial_fc(local_embeddings, local_labels,
                                  input_images=input_images)
        else:
            # Standard forward
            loss = self.partial_fc(local_embeddings, local_labels)
        return loss





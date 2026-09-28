from typing import Callable
import torch
from torch import distributed
from torch.nn.functional import linear, normalize
from losses.margin_loss import CombinedMarginLoss
from losses.adaface import AdaFaceLoss
from losses.topofr_loss import TopoFRLoss



class FC(torch.nn.Module):

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
    ):
        super(FC, self).__init__()

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_classes, embedding_size)))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
            # TopoFRLoss wraps a base margin loss; an AdaFace base still needs the
            # running norm statistics.
            needs_adaface_buffers = isinstance(margin_loss, AdaFaceLoss) or (
                isinstance(margin_loss, TopoFRLoss)
                and isinstance(margin_loss.margin_loss, AdaFaceLoss)
            )
            if needs_adaface_buffers:
                self.register_buffer('batch_mean', torch.ones(1)*(20))
                self.register_buffer('batch_std', torch.ones(1)*100)
        else:
            raise


    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        input_images: torch.Tensor = None,
    ):

        embeddings = local_embeddings
        labels = local_labels
        weight = self.weight

        norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        norm_embeddings = embeddings / norms

        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        if isinstance(self.margin_softmax, TopoFRLoss):
            # TopoFR loss: pass input images and embeddings
            result = self.margin_softmax(
                logits=logits,
                labels=labels,
                norms=norms,
                batch_mean=self.batch_mean if hasattr(self, 'batch_mean') else None,
                batch_std=self.batch_std if hasattr(self, 'batch_std') else None,
                input_images=input_images,
                embeddings=embeddings
            )

            # Handle return values
            if len(result) == 4:  # (loss, loss_dict, batch_mean, batch_std) for AdaFace base
                loss, loss_dict, batch_mean, batch_std = result
                self.batch_mean.data = batch_mean.data
                self.batch_std.data = batch_std.data
            elif len(result) == 2:  # (loss, loss_dict) for ArcFace/CosFace base
                loss, loss_dict = result
            else:
                raise ValueError(f"Unexpected return from TopoFRLoss: {len(result)} values")

        elif isinstance(self.margin_softmax, CombinedMarginLoss):
            logits = self.margin_softmax(logits=logits, labels=labels)
            loss = self.cross_entropy(logits, labels)
        elif isinstance(self.margin_softmax, AdaFaceLoss):
            logits, batch_mean, batch_std = self.margin_softmax(logits=logits, labels=labels, norms=norms,
                                                                batch_mean=self.batch_mean,
                                                                batch_std=self.batch_std)
            self.batch_mean.data = batch_mean.data
            self.batch_std.data = batch_std.data
            loss = self.cross_entropy(logits, labels)
        else:
            raise ValueError('FC margin_softmax not supported type')

        return loss




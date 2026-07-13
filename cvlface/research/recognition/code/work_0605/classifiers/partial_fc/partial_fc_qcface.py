"""Partial-FC classifier specialised for the QCFace loss.

QCFace differs from ArcFace/AdaFace in that, besides the margin-based ID loss,
it adds a magnitude-regularisation term that depends on the *ground-truth
softmax probability* (easy_probs) of each sample. Under Partial-FC the class
centres are sharded across GPUs, so this probability must be produced with a
distributed softmax that mirrors ``DistCrossEntropyFunc``.

The individual loss components (loss_id, loss_g, mean_norm) are cached as
attributes so the training loop can log them to wandb / mlflow.
"""

import torch
from torch import distributed
from torch.nn.functional import linear, normalize

from losses.qcface import QCFaceLoss
from .partial_fc import PartialFC_V2, DistCrossEntropy, AllGather


class QCFacePartialFC(PartialFC_V2):
    """Partial-FC layer that uses :class:`QCFaceLoss`.

    Recommended with ``sample_rate=1.0`` (full model parallelism) so that the
    distributed softmax used for easy_probs covers every class exactly.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        margin_loss: QCFaceLoss,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        warmup_id_only_epochs: int = 0,
    ):
        assert isinstance(margin_loss, QCFaceLoss), \
            "QCFacePartialFC requires a QCFaceLoss margin function"
        super().__init__(rank, world_size, margin_loss,
                         embedding_size, num_classes, sample_rate)
        self.dist_cross_entropy = DistCrossEntropy()

        # Two-stage warmup: during the first `warmup_id_only_epochs` epochs only
        # ID loss is used.  Once easy_probs become meaningful (classifier has
        # converged enough) norm loss is turned on.  Call set_epoch() at the
        # start of every epoch from the training loop.
        self.warmup_id_only_epochs: int = warmup_id_only_epochs
        self._current_epoch: int = 0

        # Cached scalars for logging (set on every forward).
        self._last_loss_id = 0.0
        self._last_loss_g = 0.0
        self._last_mean_norm = 0.0
        self._norm_loss_active: bool = (warmup_id_only_epochs == 0)

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of every training epoch.

        Activates norm loss once ``epoch >= warmup_id_only_epochs``.
        Prints a one-time message on the rank-0 process when norm loss is
        first switched on.
        """
        self._current_epoch = epoch
        was_active = self._norm_loss_active
        self._norm_loss_active = (epoch >= self.warmup_id_only_epochs)
        if self._norm_loss_active and not was_active and self.rank == 0:
            print(
                f"[QCFacePartialFC] epoch {epoch}: "
                f"norm loss (loss_g) activated after {self.warmup_id_only_epochs}-epoch warmup."
            )

    @torch.no_grad()
    def _dist_easy_probs(self, scaled_logits, labels, index_positive):
        """GT softmax probability per sample, gathered over all ranks.

        Args:
            scaled_logits: [N, num_local] = scale * cosine (pre-margin).
            labels:        [N, 1] local class indices, -1 for unowned samples.
            index_positive:[N, 1] bool mask of samples owned by this rank.

        Returns:
            easy_probs [N, 1] — identical on every rank.
        """
        n = scaled_logits.size(0)

        # 1. Global max per sample (numerical stability).
        global_max = scaled_logits.max(dim=1, keepdim=True).values  # [N, 1]
        distributed.all_reduce(global_max, distributed.ReduceOp.MAX)

        # 2. Global sum of exp over all classes.
        exp_logits = (scaled_logits - global_max).exp()             # [N, num_local]
        global_exp_sum = exp_logits.sum(dim=1, keepdim=True)        # [N, 1]
        distributed.all_reduce(global_exp_sum, distributed.ReduceOp.SUM)

        # 3. GT exp — only the owning rank contributes a non-zero value.
        gt_exp = torch.zeros(n, 1, device=scaled_logits.device,
                             dtype=scaled_logits.dtype)
        owned_idx = torch.where(index_positive.squeeze(1))[0]
        if owned_idx.numel() > 0:
            gt_logit = scaled_logits[owned_idx, labels[owned_idx].view(-1)]
            gt_exp[owned_idx, 0] = (gt_logit - global_max[owned_idx, 0]).exp().to(gt_exp.dtype)
        distributed.all_reduce(gt_exp, distributed.ReduceOp.SUM)

        easy_probs = gt_exp / global_exp_sum.clamp_min(1e-30)       # [N, 1]
        return easy_probs

    def forward(self, local_embeddings, local_labels):
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: "
            f"{self.last_batch_size} vs {batch_size}")

        # Gather embeddings/labels from every rank (grad-aware for embeddings).
        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size),
                        dtype=local_embeddings.dtype, device=local_embeddings.device)
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size, dtype=torch.long, device=local_labels.device)
            for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local)
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        # Raw norms carry the recognisability signal; do NOT detach embeddings.
        norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        norm_embeddings = normalize(embeddings, dim=1)
        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        # 1. easy_probs from the *pre-margin* scaled cosine (detached).
        easy_probs = self._dist_easy_probs(
            logits.detach() * self.margin_softmax.scale, labels, index_positive)

        # 2. Apply QCFace hard margin, then scale.
        margined_logits = self.margin_softmax.apply_margin(
            logits, labels, index_positive)

        # 3. Distributed CE (ID loss).
        loss_id = self.dist_cross_entropy(margined_logits, labels)

        # 4. Magnitude-regularisation loss.
        # During the warmup phase (epoch < warmup_id_only_epochs) the classifier
        # is still random, so easy_probs ≈ 0 and norm loss would collapse all
        # norms toward la=1, destroying pretrained features.  Gate it off until
        # set_epoch() indicates the warmup period is over.
        if self._norm_loss_active:
            loss_g = self.margin_softmax.compute_norm_loss(norms, easy_probs)
        else:
            loss_g = torch.zeros(1, device=loss_id.device, dtype=loss_id.dtype)

        # 5. Total.
        loss = loss_id + self.margin_softmax.lambda_g * loss_g

        # Cache scalars for logging (cheap, no extra sync beyond .item()).
        self._last_loss_id = loss_id.detach()
        self._last_loss_g = loss_g.detach()
        self._last_mean_norm = norms.mean().detach()

        return loss

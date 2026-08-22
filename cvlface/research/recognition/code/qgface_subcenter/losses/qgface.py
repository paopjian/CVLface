import math

import torch
import torch.nn.functional as F


class QGFaceLoss(torch.nn.Module):
    """QGFace contrastive loss with a proxy-updated real-time queue."""

    _QUALITY_METHODS = {"sgn", "cos", "linear"}
    _QUALITY_RANGES = {"half", "whole"}
    _QUALITY_SELECT_METHODS = {"low", "mean", "high"}

    def __init__(
        self,
        embedding_size,
        queue_size=8192,
        scale=64.0,
        margin=0.4,
        margin_method=None,
        quality_scale=0.2,
        quality_scale_method="sgn",
        quality_scale_range="half",
        quality_select_method="low",
        pair_coupling="D2N",
        mask_same_class=False,
        detach_positive=False,
        rescale=True,
        rescale_reference_size=129,
        h=0.333,
        t_alpha=0.01,
    ):
        super().__init__()
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if quality_scale_method not in self._QUALITY_METHODS:
            raise ValueError(f"Unsupported quality_scale_method: {quality_scale_method}")
        if quality_scale_range not in self._QUALITY_RANGES:
            raise ValueError(f"Unsupported quality_scale_range: {quality_scale_range}")
        if quality_select_method not in self._QUALITY_SELECT_METHODS:
            raise ValueError(f"Unsupported quality_select_method: {quality_select_method}")
        if pair_coupling not in {"S2N", "D2N"}:
            raise ValueError(f"Unsupported pair_coupling: {pair_coupling}")

        self.embedding_size = embedding_size
        self.queue_size = queue_size
        self.scale = scale
        self.margin = margin
        self.margin_method = margin_method
        self.quality_scale = quality_scale
        self.quality_scale_method = quality_scale_method
        self.quality_scale_range = quality_scale_range
        self.quality_select_method = quality_select_method
        self.pair_coupling = pair_coupling
        self.mask_same_class = mask_same_class
        self.detach_positive = detach_positive
        self.rescale = rescale
        self.rescale_reference_size = rescale_reference_size
        self.h = h
        self.t_alpha = t_alpha
        self.eps = 1e-3

        self.register_buffer("batch_mean", torch.tensor([20.0]))
        self.register_buffer("batch_std", torch.tensor([100.0]))
        self.register_buffer("key_batch_mean", torch.tensor([20.0]))
        self.register_buffer("key_batch_std", torch.tensor([100.0]))
        self.register_buffer("dynamic_margin", torch.zeros(1))
        self.register_buffer("queue", torch.zeros(queue_size, embedding_size))
        self.register_buffer("queue_proxies", torch.zeros(queue_size, embedding_size))
        self.register_buffer("queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer(
            "queue_subcenter_ids", torch.full((queue_size,), -1, dtype=torch.long)
        )
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_valid_size", torch.zeros(1, dtype=torch.long))
        self.last_metrics = {}

    @torch.no_grad()
    def enqueue(self, embeddings, labels, subcenter_ids, proxies):
        embeddings = F.normalize(embeddings.detach().float(), dim=1).to(self.queue.dtype)
        labels = labels.detach().long()
        subcenter_ids = subcenter_ids.detach().long()
        proxies = proxies.detach().float().to(self.queue_proxies.dtype)
        if embeddings.shape[0] >= self.queue_size:
            embeddings = embeddings[-self.queue_size :]
            labels = labels[-self.queue_size :]
            subcenter_ids = subcenter_ids[-self.queue_size :]
            proxies = proxies[-self.queue_size :]

        count = embeddings.shape[0]
        pointer = int(self.queue_pointer.item())
        written_indices = (
            torch.arange(count, device=labels.device, dtype=torch.long) + pointer
        ) % self.queue_size
        first_count = min(count, self.queue_size - pointer)
        self.queue[pointer : pointer + first_count].copy_(embeddings[:first_count])
        self.queue_proxies[pointer : pointer + first_count].copy_(proxies[:first_count])
        self.queue_labels[pointer : pointer + first_count].copy_(labels[:first_count])
        self.queue_subcenter_ids[pointer : pointer + first_count].copy_(
            subcenter_ids[:first_count]
        )
        remaining = count - first_count
        if remaining:
            self.queue[:remaining].copy_(embeddings[first_count:])
            self.queue_proxies[:remaining].copy_(proxies[first_count:])
            self.queue_labels[:remaining].copy_(labels[first_count:])
            self.queue_subcenter_ids[:remaining].copy_(subcenter_ids[first_count:])

        self.queue_pointer.fill_((pointer + count) % self.queue_size)
        valid_size = min(self.queue_size, int(self.queue_valid_size.item()) + count)
        self.queue_valid_size.fill_(valid_size)
        return written_indices

    @torch.no_grad()
    def get_queue(self, current_proxies=None):
        valid_size = int(self.queue_valid_size.item())
        queue = self.queue[:valid_size]
        if current_proxies is not None and valid_size:
            stored_proxies = self.queue_proxies[:valid_size]
            scale = queue.norm(p=2, dim=1, keepdim=True) / stored_proxies.norm(
                p=2, dim=1, keepdim=True
            ).clamp_min(self.eps)
            queue = queue + scale * (current_proxies.float() - stored_proxies)
        return (
            queue,
            self.queue_labels[:valid_size],
            self.queue_subcenter_ids[:valid_size],
        )

    @torch.no_grad()
    def _margin_scalers(self, q_norms, k_norms, quality_stat_norms=None):
        q_norms = q_norms.detach().float().clamp(0.001, 100.0)
        k_norms = k_norms.detach().float().clamp(0.001, 100.0)
        if quality_stat_norms is None:
            q_stat_norms, k_stat_norms = q_norms, k_norms
        else:
            q_stat_norms, k_stat_norms = quality_stat_norms
            q_stat_norms = q_stat_norms.detach().float().clamp(0.001, 100.0)
            k_stat_norms = k_stat_norms.detach().float().clamp(0.001, 100.0)

        def update(mean_buffer, std_buffer, norms):
            mean_buffer.mul_(1 - self.t_alpha).add_(norms.mean() * self.t_alpha)
            std_buffer.mul_(1 - self.t_alpha).add_(
                norms.std(unbiased=False) * self.t_alpha
            )

        update(self.batch_mean, self.batch_std, q_stat_norms)
        update(self.key_batch_mean, self.key_batch_std, k_stat_norms)

        def scale(norms, mean, std):
            scaler = (norms - mean) / (std + self.eps)
            return (scaler * self.h).clamp(-1, 1)

        return (
            scale(q_norms, self.batch_mean, self.batch_std),
            scale(k_norms, self.key_batch_mean, self.key_batch_std),
        )

    def _quality_weights(
        self,
        q_norms,
        k_norms,
        quality_stat_norms=None,
        margin_scalers=None,
    ):
        if margin_scalers is None:
            q_scaler, k_scaler = self._margin_scalers(
                q_norms, k_norms, quality_stat_norms
            )
        else:
            q_scaler, k_scaler = margin_scalers
        q_normalized = (q_scaler + 1) / 2
        k_normalized = (k_scaler + 1) / 2
        if self.quality_select_method == "low":
            selected = torch.minimum(q_normalized, k_normalized)
            selected_norms = torch.minimum(q_norms, k_norms)
        elif self.quality_select_method == "mean":
            selected = (q_normalized + k_normalized) / 2
            selected_norms = (q_norms + k_norms) / 2
        else:
            selected = torch.maximum(q_normalized, k_normalized)
            selected_norms = torch.maximum(q_norms, k_norms)

        if self.quality_scale == 0:
            weights = torch.ones_like(selected)
        elif self.quality_scale_method == "sgn":
            high_weight = 1.0 if self.quality_scale_range == "whole" else 0.0
            weights = torch.where(
                selected < self.quality_scale,
                torch.ones_like(selected),
                torch.full_like(selected, high_weight),
            )
        elif self.quality_scale_method == "cos":
            low_weight = (
                1 - torch.cos(math.pi * selected / (self.quality_scale + self.eps))
            ) / 2
            if self.quality_scale_range == "whole":
                high_weight = (
                    torch.cos(
                        math.pi
                        * (selected - self.quality_scale + (1 - self.quality_scale))
                        / (2 * (1 - self.quality_scale + self.eps))
                    )
                    + 1
                )
            else:
                high_weight = torch.zeros_like(selected)
            weights = torch.where(selected < self.quality_scale, low_weight, high_weight)
        else:
            low_weight = selected / (self.quality_scale + self.eps)
            if self.quality_scale_range == "whole":
                high_weight = (selected - 1) / (self.quality_scale - 1 + self.eps)
            else:
                high_weight = torch.zeros_like(selected)
            weights = torch.where(selected < self.quality_scale, low_weight, high_weight)

        return weights, selected_norms

    def _positive_margin(self, cosine, positive_labels):
        if self.margin_method is None:
            return cosine

        rows = torch.arange(cosine.shape[0], device=cosine.device)
        positive = cosine[rows, positive_labels]
        negative_view = cosine.detach().clone()
        negative_view[rows, positive_labels] = -1
        hardest_negative = negative_view.max(dim=1).values
        batch_margin = (positive.detach() - hardest_negative).mean()
        self.dynamic_margin.mul_(1 - self.t_alpha).add_(batch_margin * self.t_alpha)

        if self.margin_method == "dynamic":
            margin = self.dynamic_margin.to(cosine.dtype)
        elif self.margin_method == "static":
            margin = self.margin
        else:
            raise ValueError(f"Unsupported margin_method: {self.margin_method}")
        cosine = cosine.clone()
        cosine[rows, positive_labels] -= margin
        return cosine

    def forward(
        self,
        q_embeddings,
        k_embeddings,
        q_norms,
        k_norms,
        query_labels,
        negative_embeddings,
        negative_labels,
        positive_indices,
        quality_stat_norms=None,
        margin_scalers=None,
    ):
        if negative_embeddings.shape[0] == 0:
            return q_embeddings.sum() * 0

        q = F.normalize(q_embeddings.float(), dim=1)
        k = F.normalize(k_embeddings.float(), dim=1)
        negatives = F.normalize(negative_embeddings.detach().float(), dim=1)
        positive_key = k.detach() if self.detach_positive else k
        positive = (q * positive_key).sum(dim=1, keepdim=True)
        negative = q @ negatives.t()

        query_labels_for_mask = query_labels
        positive_indices_for_mask = positive_indices
        if self.pair_coupling == "D2N":
            reverse_key = q.detach() if self.detach_positive else q
            reverse_positive = (k * reverse_key).sum(dim=1, keepdim=True)
            positive = torch.cat([positive, reverse_positive], dim=0)
            negative = torch.cat([negative, k @ negatives.t()], dim=0)
            query_labels_for_mask = torch.cat([query_labels, query_labels], dim=0)

        if self.mask_same_class:
            mask = query_labels_for_mask.view(-1, 1) == negative_labels.view(1, -1)
        else:
            mask = torch.zeros_like(negative, dtype=torch.bool)
            if positive_indices_for_mask.ndim == 1:
                positive_indices_for_mask = positive_indices_for_mask[:, None]
            if self.pair_coupling == "D2N":
                positive_indices_for_mask = torch.cat(
                    [positive_indices_for_mask, positive_indices_for_mask], dim=0
                )
            rows = torch.arange(mask.shape[0], device=mask.device)[:, None]
            valid = (positive_indices_for_mask >= 0) & (
                positive_indices_for_mask < mask.shape[1]
            )
            mask[
                rows.expand_as(positive_indices_for_mask)[valid],
                positive_indices_for_mask[valid],
            ] = True
        negative = negative.masked_fill(mask, 0)

        cosine = torch.cat([positive, negative], dim=1).clamp(-1, 1)
        positive_labels = torch.zeros(cosine.shape[0], dtype=torch.long, device=cosine.device)
        cosine = self._positive_margin(cosine, positive_labels)

        quality_weights, selected_norms = self._quality_weights(
            q_norms,
            k_norms,
            quality_stat_norms,
            margin_scalers,
        )
        if self.pair_coupling == "D2N":
            quality_weights = torch.cat([quality_weights, quality_weights], dim=0)
        logits = cosine * self.scale * quality_weights.to(cosine.dtype)
        loss = F.cross_entropy(logits, positive_labels)
        if self.rescale and cosine.shape[1] > 1:
            loss = loss / math.log(cosine.shape[1]) * math.log(self.rescale_reference_size)

        self.last_metrics = {
            "qgface/contrastive_loss": loss.detach(),
            "qgface/positive_similarity": positive.detach().mean(),
            "qgface/quality_weight": quality_weights.detach().mean(),
            "qgface/selected_norm": selected_norms.detach().mean(),
            "qgface/queue_size": torch.tensor(
                negative_embeddings.shape[0], device=loss.device, dtype=torch.float32
            ),
        }
        return loss

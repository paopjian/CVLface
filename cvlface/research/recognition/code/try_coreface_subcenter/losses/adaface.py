import torch
import torch.nn.functional as F
import math


class AdaFaceLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m,
                 h,
                 t_alpha,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.interclass_filtering_threshold = interclass_filtering_threshold
        self.eps = 1e-3

    def forward(self, logits, labels, norms, batch_mean, batch_std):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * batch_mean
            batch_std = std * self.t_alpha + (1 - self.t_alpha) * batch_std

        margin_scaler = (safe_norms - batch_mean) / (batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1).view(-1)
        margin_scaler = margin_scaler[index_positive]

        target_logit = logits[index_positive, labels[index_positive].view(-1)]


        #########
        with torch.no_grad():
            # g_angular
            target_logit.arccos_()
            margin_final_logit = target_logit + (self.m * margin_scaler * -1)
            margin_final_logit.cos_()
            # g_additive
            margin_final_logit = margin_final_logit - (self.m + (self.m * margin_scaler))
            # make margin_final_logit as same dtype as logits
            margin_final_logit = margin_final_logit.type(logits.dtype)
            logits[index_positive, labels[index_positive].view(-1)] = margin_final_logit

        # scale
        logits = logits * self.s

        return logits, batch_mean, batch_std


class ContraFaceLoss(torch.nn.Module):
    """Sample-guided contrastive loss used by CoreFace.

    The diagonal pairs are the two dropout views of the same sample.  Other
    samples of the same identity are excluded from the negative pool, which
    avoids penalizing legitimate positive pairs in face batches.
    """

    def __init__(self, scale=64.0, margin_momentum=0.99):
        super().__init__()
        self.scale = scale
        self.margin_momentum = margin_momentum
        self.margin = 0.0

    def forward(self, feature1, feature2, labels):
        if feature1.shape != feature2.shape:
            raise ValueError(f'feature shapes must match, got {feature1.shape} and {feature2.shape}')
        batch_size = feature1.shape[0]
        if batch_size == 0:
            return feature1.sum() * 0.0

        feature1 = F.normalize(feature1.float(), dim=1)
        feature2 = F.normalize(feature2.float(), dim=1)
        similarity = feature1 @ feature2.transpose(0, 1)
        targets = torch.arange(batch_size, device=similarity.device)

        # Do not treat another image of the same identity as a negative.
        labels = labels.reshape(-1)
        same_identity = labels[:, None].eq(labels[None, :])
        negative_mask = same_identity.clone()
        negative_mask.fill_diagonal_(True)
        negatives = similarity.masked_fill(negative_mask, float('-inf'))
        hardest_negative = negatives.max(dim=1).values
        positive = similarity.diagonal()
        valid_margin = torch.isfinite(hardest_negative)
        if valid_margin.any():
            batch_margin = (positive.detach()[valid_margin] - hardest_negative.detach()[valid_margin]).mean()
            # CoreFace follows the reference implementation: the current
            # batch provides 99% of the adaptive margin and the previous
            # value contributes the remaining 1%.
            self.margin = (self.margin_momentum * batch_margin.item()
                           + (1.0 - self.margin_momentum) * self.margin)

        logits = similarity * self.scale
        logits[targets, targets] -= self.scale * self.margin
        return F.cross_entropy(logits, targets)


# Keep the name used by the reference CoreFace implementation available to
# experiments that import the criterion directly.
ContraFace = ContraFaceLoss

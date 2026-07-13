"""QCFace margin loss + magnitude-regularisation loss.

Reference: "QCFace: Image Quality Control for boosting Face Representation & Recognition"
           WACV 2026 — Doan-Ngo et al.

This module is analogous to AdaFaceLoss in the CVLface framework:
  - apply_margin() modifies the GT cosine logit (hard ArcFace-style margin
    with fallback: if cos(θ) ≤ 0 keep the original cosine).
  - compute_norm_loss() computes the magnitude-regularisation term L_qc that
    pushes each sample's embedding norm toward the analytically optimal value
    z*(p) where p is the GT softmax probability (easy_probs).

The caller (partial_fc_qcface.py) is responsible for:
  1. Computing easy_probs via a distributed softmax (pre-margin).
  2. Calling apply_margin().
  3. Running DistCrossEntropy on the margined, scaled logits → loss_id.
  4. Calling compute_norm_loss(norms, easy_probs) → loss_g.
  5. Combining: total_loss = loss_id + lambda_g * loss_g.
"""

import math
import torch
import torch.nn as nn
from sympy import Symbol, sqrt as sym_sqrt, solve


def _solve_k(ua: float, la: float) -> float:
    """Solve for the optimal constant k such that z*(0.5) = (ua + la) / 2."""
    k = Symbol('k')
    z_half = sym_sqrt(((k - 1) * 0.5 + 1) /
                      ((k / (ua ** 2) - 1 / (la ** 2)) * 0.5 + 1 / (la ** 2)))
    target = (ua + la) / 2.0
    solutions = solve(z_half - target, k)
    return float(str(solutions[0]))


class QCFaceLoss(nn.Module):
    """QCFace hard-margin loss (= ArcFace margin) + magnitude-regularisation.

    Args:
        scale: Logit scaling factor s (default 64).
        m:     Angular margin in radians (default 0.5).
        ua:    Upper bound of the allowed norm range (default 100).
        la:    Lower bound of the allowed norm range (default 1).
        lambda_g: Weight for the magnitude-regularisation term (default 1.0).
        eps:   Numerical stability epsilon (default 1e-3).
    """

    def __init__(
        self,
        scale: float = 64.0,
        m: float = 0.5,
        ua: float = 100.0,
        la: float = 1.0,
        lambda_g: float = 1.0,
        interclass_filtering_threshold: float = 0,
        warmup_id_only_epochs: int = 0,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.scale = scale
        self.m = m
        self.ua = ua
        self.la = la
        self.lambda_g = lambda_g
        self.interclass_filtering_threshold = interclass_filtering_threshold
        self.warmup_id_only_epochs = warmup_id_only_epochs
        self.eps = eps

        # Pre-compute trig constants for the margin.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Solve for the k constant once at construction (uses sympy).
        self.k = _solve_k(ua, la)

    # ------------------------------------------------------------------
    # Internal helpers (mirrors QCFace original repo)
    # ------------------------------------------------------------------

    def _qcfnc(
        self,
        n: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """The QC energy function f(n, p) used to derive the norm loss."""
        ua, la, k, eps = self.ua, self.la, self.k, self.eps
        return (k * p * (1.0 / (n + eps) + n / (ua ** 2))
                + (1.0 - p) * (1.0 / (n + eps) + n / (la ** 2)))

    def _optimal_norm(self, p: torch.Tensor) -> torch.Tensor:
        """Compute the analytically optimal norm z*(p)."""
        ua, la, k, eps = self.ua, self.la, self.k, self.eps
        num = (k - 1) * p + 1.0
        den = (k / (ua ** 2) - 1.0 / (la ** 2)) * p + 1.0 / (la ** 2) + eps
        return torch.sqrt(num / den)

    # ------------------------------------------------------------------
    # Public interface called by the classifier
    # ------------------------------------------------------------------

    def apply_margin(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        index_positive: torch.Tensor,
    ) -> torch.Tensor:
        """Apply QCFace hard margin to GT logit entries and scale.

        Args:
            logits: [N, num_local] cosine similarities, clamped to [-1, 1].
                    Only the columns owned by this GPU rank are present.
            labels: [N, 1] local class indices; -1 for unowned samples.
            index_positive: [N, 1] bool mask — True when this rank owns the
                            GT class of that sample.

        Returns:
            Scaled logits [N, num_local] with the GT entry replaced by
            cos(θ + m) (or cos(θ) if θ > π/2).
        """
        # Optionally filter inter-class logits.
        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = (logits > self.interclass_filtering_threshold).float()
                owned_idx = torch.where(index_positive.squeeze(1))[0]
                mask = torch.ones_like(logits[owned_idx])
                mask.scatter_(1, labels[owned_idx], 0)
                dirty[owned_idx] *= mask
                logits = (1 - dirty) * logits

        owned_idx = torch.where(index_positive.squeeze(1))[0]
        if owned_idx.numel() > 0:
            gt_cos = logits[owned_idx, labels[owned_idx].view(-1)].float()  # fp32 for precision
            sin_theta = torch.sqrt((1.0 - gt_cos.pow(2)).clamp(min=0.0))
            gt_cos_m = gt_cos * self.cos_m - sin_theta * self.sin_m
            # Fallback: keep original cosine when θ > π/2.
            gt_cos_m = torch.where(gt_cos > 0.0, gt_cos_m, gt_cos)
            logits[owned_idx, labels[owned_idx].view(-1)] = gt_cos_m.to(logits.dtype)

        return logits * self.scale

    def compute_norm_loss(
        self,
        norms: torch.Tensor,
        easy_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the magnitude-regularisation loss L_qc.

        Args:
            norms:      [N, 1] L2 norms of the raw (un-normalised) embeddings.
            easy_probs: [N, 1] GT softmax probabilities (computed pre-margin,
                        detached, gathered from all ranks).

        Returns:
            Scalar norm loss (mean over the batch).
        """
        with torch.no_grad():
            opt_n = self._optimal_norm(easy_probs)          # [N, 1]
            opt_loss = self._qcfnc(opt_n, easy_probs)       # [N, 1]

        norm_loss = self._qcfnc(norms, easy_probs) - opt_loss  # [N, 1]
        return norm_loss.mean()

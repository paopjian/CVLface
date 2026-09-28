"""
TopoFR Loss: Combining classification loss with topological loss
Based on: TopoFR: A Closer Look at Topology Alignment on Face Recognition (NeurIPS 2024)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import distributed
import numpy as np
from .topology import compute_topological_loss
from .gum import gauss_unif
from .adaface import AdaFaceLoss


def Entropy(input_):
    """Compute entropy for softmax predictions"""
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class DistCrossEntropyPerSampleFunc(torch.autograd.Function):
    """PartialFC(模型并行)下的 per-sample 交叉熵.

    PartialFC 每个 rank 只持有部分类中心, 非本地类的 label 被置为 -1.
    因此不能用 nn.CrossEntropyLoss: (1) target=-1 越界会触发 device-side assert;
    (2) softmax 分母必须跨 rank all_reduce, 只在本地类上归一化是错的.

    与 DistCrossEntropy 的区别: 返回 per-sample loss 而不是 mean,
    因为 TopoFR 的 GUM 加权需要逐样本的损失. 同时顺带返回全局 entropy 和
    真值概率, 避免 TopoFRLoss 里再对 margin_logits 做一次(错误的)本地 softmax.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        batch_size = logits.size(0)
        # 数值稳定: 全局最大值
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)  # 此后 logits 存的是全局 softmax 概率(仅本地类列)

        index = torch.where(label != -1)[0]
        prob_gt = torch.zeros(batch_size, 1, device=logits.device, dtype=logits.dtype)
        prob_gt[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(prob_gt, distributed.ReduceOp.SUM)

        # 全局 entropy: 本地类求和后跨 rank 相加
        eps = 1e-5
        entropy = torch.sum(-logits * torch.log(logits + eps), dim=1)
        distributed.all_reduce(entropy, distributed.ReduceOp.SUM)

        ctx.save_for_backward(index, logits, label)
        prob_gt_1d = prob_gt.squeeze(1)
        loss_per_sample = prob_gt_1d.clamp_min(1e-30).log().mul(-1)
        ctx.mark_non_differentiable(entropy, prob_gt_1d)
        return loss_per_sample, entropy, prob_gt_1d

    @staticmethod
    def backward(ctx, grad_loss, grad_entropy, grad_prob):
        index, logits, label = ctx.saved_tensors
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device, dtype=logits.dtype
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        # d(-log p_y)/d logit_j = p_j - onehot_j; 下游 .mean() 已把 1/batch 带进 grad_loss
        return logits * grad_loss.unsqueeze(1), None


class TopoFRLoss(nn.Module):
    """
    TopoFR Loss: ArcFace + Topological Loss + GUM-based sample weighting

    Args:
        margin_loss: Base margin loss (e.g., ArcFace, CosFace)
        topo_weight: Weight for topological loss (default: 0.1)
        use_gum: Whether to use GUM model for sample weighting (default: True)
        temp: Temperature for GUM weighting (default: 1)
    """

    def __init__(self,
                 margin_loss,
                 topo_weight=0.1,
                 use_gum=True,
                 temp=1):
        super().__init__()
        self.margin_loss = margin_loss
        self.topo_weight = topo_weight
        self.use_gum = use_gum
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # 诊断用: 设 TOPOFR_LOG_EVERY=N 时, rank0 每 N 步打印一次损失配比.
        # total loss 里 loss_cls 和 topo_weight*loss_topo 谁占主导, 不看是黑箱.
        self._step = 0
        self._log_every = int(os.environ.get('TOPOFR_LOG_EVERY', 0))

    def forward(self, logits, labels, norms=None, batch_mean=None, batch_std=None,
                input_images=None, embeddings=None, model_parallel=False):
        """
        Forward pass for TopoFR loss

        Args:
            logits: Classifier logits (before margin)
            labels: Ground truth labels
            norms: Feature norms (for AdaFace)
            batch_mean: Running mean of norms (for AdaFace)
            batch_std: Running std of norms (for AdaFace)
            input_images: Original input images for topological loss
            embeddings: Feature embeddings for topological loss
            model_parallel: True 时走 PartialFC 的分布式交叉熵(logits 只含本地类,
                非本地类 label 为 -1). False 时走普通 FC 的本地交叉熵.

        Returns:
            loss: Total loss
            loss_dict: Dictionary containing loss components
        """
        # Apply margin loss. AdaFace/ArcFace 都接受 PartialFC 的 2D labels
        # (内部靠 gather/scatter 用, 需要 (N,1) 形状), 所以原样传进去.
        is_adaface = isinstance(self.margin_loss, AdaFaceLoss)
        if is_adaface:
            margin_logits, batch_mean, batch_std = self.margin_loss(
                logits, labels, norms, batch_mean, batch_std)
        else:
            # Standard margin loss (ArcFace, CosFace)
            margin_logits = self.margin_loss(logits, labels)

        if model_parallel:
            # PartialFC: logits 只含本地类, 非本地样本 label=-1.
            # per-sample loss / entropy / 真值概率都要跨 rank 归约才正确.
            labels_2d = labels if labels.dim() > 1 else labels.view(-1, 1)
            loss_cls_sample, entropy, probability_gt = \
                DistCrossEntropyPerSampleFunc.apply(margin_logits, labels_2d)
        else:
            # 普通 FC: logits 覆盖全部类, 本地算即可. CrossEntropyLoss 要 1D labels.
            labels_1d = labels.squeeze() if labels.dim() > 1 else labels
            loss_cls_sample = self.criterion(margin_logits, labels_1d)
            entropy = None
            probability_gt = None

        # Sample weighting with GUM model
        if self.use_gum:
            with torch.no_grad():
                if entropy is None:
                    # 非模型并行: 本地 softmax 即全局 softmax
                    softmax_probs = torch.nn.Softmax(dim=1)(margin_logits)
                    entropy = Entropy(softmax_probs)
                    probability_gt = softmax_probs.gather(
                        1, labels_1d.view(-1, 1)).squeeze(1)

                entropy_np = entropy.float().cpu().detach().numpy()

                # Alternate sign for paired samples (if using paired augmentation)
                # In standard training, this just processes entropy as-is
                entropy_np[2*np.arange(len(entropy_np)//2)] = -1 * entropy_np[2*np.arange(len(entropy_np)//2)]

                # GUM model for sample weighting
                sample_weight, GUM_pi, GUM_sigma = gauss_unif(entropy_np.reshape(-1, 1))
                sample_weight = torch.tensor(sample_weight, device=logits.device, dtype=logits.dtype)
                probability_gt = probability_gt.to(logits.dtype)

            # Combined weighting
            w1 = torch.pow((2 - sample_weight), self.temp)
            w2 = (1 - probability_gt)
            loss_cls = (w1 * w2 * loss_cls_sample).mean()
        else:
            loss_cls = loss_cls_sample.mean()

        # Topological loss
        loss_topo = 0.0
        if self.topo_weight > 0 and input_images is not None and embeddings is not None:
            loss_topo = compute_topological_loss(input_images, embeddings, use_grad=True)
            total_loss = loss_cls + self.topo_weight * loss_topo
        else:
            total_loss = loss_cls

        # Prepare loss dictionary for logging
        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_topo': loss_topo.item() if isinstance(loss_topo, torch.Tensor) else loss_topo,
            'loss_total': total_loss.item(),
        }

        if self.use_gum:
            loss_dict['gum_pi'] = GUM_pi
            loss_dict['gum_sigma'] = GUM_sigma

        self._step += 1
        if self._log_every and self._step % self._log_every == 0 \
                and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[TopoFR] step={self._step} "
                  f"loss_cls={loss_dict['loss_cls']:.4f} "
                  f"loss_topo={loss_dict['loss_topo']:.4f} "
                  f"topo_weighted={self.topo_weight * loss_dict['loss_topo']:.4f} "
                  f"total={loss_dict['loss_total']:.4f}", flush=True)

        # Return batch_mean and batch_std for AdaFace
        if is_adaface:
            return total_loss, loss_dict, batch_mean, batch_std
        else:
            return total_loss, loss_dict

from typing import Callable
import math
import torch
from torch import distributed
from torch.nn.functional import linear, normalize
from losses.margin_loss import CombinedMarginLoss
from losses.adaface import AdaFaceLoss



class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 2

    def __init__(
        self,
        rank: int,
        world_size: int,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        num_subcenters: int = 1,
        ohem_ratio: float = 1.0,
        ohem_min_kept: int = 0,
        ohem_warmup_steps: int = 0,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        num_subcenters: int
            Number of prototypes maintained for each identity class.
        ohem_ratio: float
            OHEM 每步保留的困难样本比例 (按 per-sample loss top-k), 1.0 = 关闭。
        ohem_min_kept: int
            OHEM 保留样本数的下限 (小 batch/调试防退化)。
        ohem_warmup_steps: int
            有效 ratio 从 1.0 线性退火到 ohem_ratio 的步数, 0 = 立即生效。
        """
        super(PartialFC_V2, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = rank
        self.world_size = world_size

        self.dist_cross_entropy = DistCrossEntropy(
            ohem_ratio=ohem_ratio,
            ohem_min_kept=ohem_min_kept,
            ohem_warmup_steps=ohem_warmup_steps,
        )
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        if num_subcenters < 1:
            raise ValueError(f"num_subcenters must be at least 1, got {num_subcenters}")
        self.num_subcenters = int(num_subcenters)
        self.actual_num_classes = num_classes

        # make num_class divisible by self.world_size for ddp
        _num_classes = num_classes // self.world_size * self.world_size
        if _num_classes < num_classes:
            _num_classes = _num_classes + self.world_size
        num_classes = _num_classes
        self.num_classes: int = num_classes

        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )

        # for i in range(8):
        #     num_local = (num_classes // self.world_size + int( i < num_classes % self.world_size ))
        #     class_start = num_classes // self.world_size * i + min( i, num_classes % self.world_size )
        #     print(num_local, class_start)

        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(torch.normal(
            0,
            0.01,
            (self.num_local * self.num_subcenters, embedding_size),
        ))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
            if isinstance(margin_loss, AdaFaceLoss):
                self.register_buffer('batch_mean', torch.ones(1)*(20))
                self.register_buffer('batch_std', torch.ones(1)*100)
        else:
            raise

    def sample(self, labels, index_positive):
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=labels.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        subcenter_offsets = torch.arange(self.num_subcenters, device=index.device)
        self.subcenter_weight_index = (
            index[:, None] * self.num_subcenters + subcenter_offsets[None, :]
        ).reshape(-1)
        return self.weight[self.subcenter_weight_index]

    def compute_logits(self, embeddings, weight):
        """Compute one class logit as the maximum over its subcenters."""
        if weight.size(0) % self.num_subcenters != 0:
            raise ValueError(
                f"weight rows ({weight.size(0)}) must be divisible by "
                f"num_subcenters ({self.num_subcenters})"
            )

        norm_embeddings = normalize(embeddings, dim=1)
        norm_weight_activated = normalize(weight, dim=1)
        logits = linear(norm_embeddings, norm_weight_activated)
        if self.num_subcenters > 1:
            num_active_classes = weight.size(0) // self.num_subcenters
            logits = logits.reshape(
                embeddings.size(0), num_active_classes, self.num_subcenters
            ).amax(dim=2).clone()
        return logits

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """

        local_labels = local_labels.reshape(-1).long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size), dtype=local_embeddings.dtype, device=local_embeddings.device)
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size, dtype=torch.long, device=local_labels.device) for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        # with torch.cuda.amp.autocast(self.fp16):
        norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        logits = self.compute_logits(embeddings, weight)

        if isinstance(self.margin_softmax, CombinedMarginLoss):
            logits = self.margin_softmax(logits=logits, labels=labels)
        elif isinstance(self.margin_softmax, AdaFaceLoss):
            logits, batch_mean, batch_std = self.margin_softmax(logits=logits, labels=labels, norms=norms,
                                                                batch_mean=self.batch_mean,
                                                                batch_std=self.batch_std)
            self.batch_mean.data = batch_mean.data
            self.batch_std.data = batch_std.data
        else:
            raise ValueError('parital FC margin_softmax not supported type')

        loss = self.dist_cross_entropy(logits, labels)
        return loss


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):

    可选 OHEM (online hard example mining, 分类版): 每步对 per-sample NLL 取
    top-k 困难样本, loss = mean(nll[kept]), k = ceil(ratio_eff * B)。
    - ratio_eff >= 1 时 k == B, 走与原实现完全一致的路径 (行为零差异)。
    - forward 里 per-sample loss 向量经 all_reduce 后各 rank 位相同,
      topk 选择天然跨 rank 一致, 无需额外通信。
    - backward 的手工梯度需与 forward 的聚合方式对齐: 选中行乘 1/k,
      未选中行梯度置 0 (该样本对 loss 无贡献, 含 softmax 分母项)。
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor,
                ohem_ratio_eff: float, ohem_min_kept: int):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device, dtype=logits.dtype)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        log_prob = loss.clamp_min_(1e-30).log_()  # per-sample log p_true (B,1), 各 rank 位相同

        # ---------- OHEM 困难样本选择 ----------
        ohem_weight = None
        kept = batch_size
        if ohem_ratio_eff < 1.0:
            with torch.no_grad():
                kept = math.ceil(ohem_ratio_eff * batch_size)
                kept = max(kept, min(int(ohem_min_kept), batch_size), 1)
                kept = min(kept, batch_size)
            if kept < batch_size:
                with torch.no_grad():
                    # log_prob 最小 = p_true 最小 = NLL 最大 = 最难样本
                    hard_idx = torch.topk(log_prob.flatten(), k=kept,
                                          largest=False, sorted=False).indices
                    ohem_weight = torch.zeros_like(log_prob)
                    ohem_weight[hard_idx] = 1.0 / kept
                    # 与 backward 里 logits 的 in-place 乘法对齐 dtype (bf16 下避免类型提升报错)
                    ohem_weight = ohem_weight.to(logits.dtype)

        if ohem_weight is None:
            out_loss = log_prob.mean() * (-1)
        else:
            out_loss = (log_prob.float() * ohem_weight.float()).sum() * (-1)

        ctx.ohem_weight = ohem_weight
        ctx.save_for_backward(index, logits, label)

        # 统计 [全批 NLL 均值, 选中 NLL 均值, kept, batch] — 张量返回, 日志时才 .item(), 避免每步同步
        with torch.no_grad():
            stats = torch.stack([
                -log_prob.detach().float().mean(),
                out_loss.detach().float(),
                torch.full((), float(kept), device=logits.device),
                torch.full((), float(batch_size), device=logits.device),
            ])
        return out_loss, stats

    @staticmethod
    def backward(ctx, loss_gradient, stats_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        ohem_weight = ctx.ohem_weight
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        if ohem_weight is not None:
            # OHEM: 行级权重 (选中=1/k, 未选中=0), 与 forward 的 mean-over-kept 对齐
            logits.mul_(ohem_weight)
        else:
            logits.div_(logits.size(0))
        return logits * loss_gradient, None, None, None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self, ohem_ratio: float = 1.0, ohem_min_kept: int = 0,
                 ohem_warmup_steps: int = 0):
        super(DistCrossEntropy, self).__init__()
        if not (0.0 < ohem_ratio <= 1.0):
            raise ValueError(f"ohem_ratio must be in (0, 1], got {ohem_ratio}")
        self.ohem_ratio = float(ohem_ratio)
        self.ohem_min_kept = int(ohem_min_kept)
        self.ohem_warmup_steps = int(ohem_warmup_steps)
        # 随 classifier state_dict 存取; 断点续训/跨阶段加载 classifier 时保持退火进度
        self.register_buffer('ohem_step', torch.zeros(1, dtype=torch.long))
        self._ohem_step_cpu = 0
        # 上一步统计, log_classifier 读取; 张量懒取值
        self.last_stats = None
        self.last_ratio_eff = 1.0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        key = prefix + 'ohem_step'
        if key in state_dict:
            # state_dict 张量通常在 CPU 上, 此处一次性回读, 之后 warmup 判断走 cpu 计数免同步
            self._ohem_step_cpu = int(state_dict[key].flatten()[0].item())

    def forward(self, logit_part, label_part):
        # warmup: 有效 ratio 从 1.0 线性退火到 ohem_ratio, 分类器冷启动阶段先按标准 CE 走
        if self.ohem_warmup_steps > 0 and self.ohem_ratio < 1.0:
            progress = min(self._ohem_step_cpu / self.ohem_warmup_steps, 1.0)
            ratio_eff = 1.0 + (self.ohem_ratio - 1.0) * progress
        else:
            ratio_eff = self.ohem_ratio
        self._ohem_step_cpu += 1
        self.ohem_step.fill_(self._ohem_step_cpu)
        loss, stats = DistCrossEntropyFunc.apply(
            logit_part, label_part, ratio_eff, self.ohem_min_kept)
        self.last_stats = stats
        self.last_ratio_eff = ratio_eff
        return loss


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply

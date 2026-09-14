"""校验 DistCrossEntropyPerSampleFunc 的正确性.

用 world_size=1 (all_reduce 退化为恒等) 对比三方:
  1. 我的 per-sample 分布式 CE
  2. 参考实现 DistCrossEntropy (返回 mean)
  3. torch 原生 F.cross_entropy
再单独校验梯度.
"""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29517')
dist.init_process_group('gloo', rank=0, world_size=1)

import sys
sys.path.insert(0, '/root/zhaokj/CVLface/cvlface')
sys.path.insert(0, '/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr')
from losses.topofr_loss import DistCrossEntropyPerSampleFunc
from classifiers.partial_fc.partial_fc import DistCrossEntropy

torch.manual_seed(0)
N, C = 8, 20
base = torch.randn(N, C, dtype=torch.float64)
labels = torch.randint(0, C, (N, 1))

# ---- 1. 前向: per-sample loss vs 原生 CE ----
a = base.clone().requires_grad_(True)
loss_ps, entropy, prob_gt = DistCrossEntropyPerSampleFunc.apply(a, labels)

ref_ce = F.cross_entropy(base, labels.squeeze(1), reduction='none')
print('per-sample loss  max|diff| vs F.cross_entropy :', (loss_ps - ref_ce).abs().max().item())

# ---- 2. mean 与参考 DistCrossEntropy 一致 ----
b = base.clone().requires_grad_(True)
ref_mean = DistCrossEntropy()(b, labels)
print('mean loss        |diff| vs DistCrossEntropy   :', (loss_ps.mean() - ref_mean).abs().item())

# ---- 3. entropy / prob_gt ----
p = torch.softmax(base, dim=1)
ref_ent = torch.sum(-p * torch.log(p + 1e-5), dim=1)
ref_pgt = p.gather(1, labels).squeeze(1)
print('entropy          max|diff|                    :', (entropy - ref_ent).abs().max().item())
print('prob_gt          max|diff|                    :', (prob_gt - ref_pgt).abs().max().item())

# ---- 4. 梯度: 对 .mean() 求导, 应等于原生 CE mean 的梯度 ----
loss_ps.mean().backward()
c = base.clone().requires_grad_(True)
F.cross_entropy(c, labels.squeeze(1), reduction='mean').backward()
print('grad(mean)       max|diff| vs F.cross_entropy :', (a.grad - c.grad).abs().max().item())

# ---- 5. 梯度: 带 per-sample 权重 (GUM 的实际用法) ----
w = torch.rand(N, dtype=torch.float64)
d = base.clone().requires_grad_(True)
l2, _, _ = DistCrossEntropyPerSampleFunc.apply(d, labels)
(w * l2).mean().backward()
e = base.clone().requires_grad_(True)
(w * F.cross_entropy(e, labels.squeeze(1), reduction='none')).mean().backward()
print('grad(weighted)   max|diff| vs F.cross_entropy :', (d.grad - e.grad).abs().max().item())

# ---- 6. label=-1 (非本地类) 不崩, 且该行梯度为纯概率 ----
labels_masked = labels.clone()
labels_masked[0] = -1
labels_masked[3] = -1
f = base.clone().requires_grad_(True)
l3, ent3, pgt3 = DistCrossEntropyPerSampleFunc.apply(f, labels_masked)
l3.mean().backward()
print()
print('label=-1 未崩溃. prob_gt[masked] =', pgt3[[0, 3]].tolist(), '(应为 0)')
row0_expect = torch.softmax(base, dim=1)[0] / N
print('masked 行梯度 max|diff| vs p/N              :', (f.grad[0] - row0_expect).abs().max().item())

dist.destroy_process_group()

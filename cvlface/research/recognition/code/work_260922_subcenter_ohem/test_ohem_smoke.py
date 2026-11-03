"""OHEM DistCrossEntropy 冒烟测试 (单进程 gloo, 不依赖数据集/GPU)。

验证点:
  1. ohem_ratio=1.0 (默认): loss 与 F.cross_entropy 一致, 梯度与 autograd 一致
     —— 即不开启 OHEM 时行为与原实现零差异。
  2. ohem_ratio<1.0: loss == top-k per-sample NLL 均值; 梯度与 autograd 参考
     (softmax CE 手写 + 选中行 mean) 一致, 未选中行梯度为 0。
  3. warmup: 计数从 0 起 ratio_eff=1.0, 越过 warmup_steps 后收敛到 ohem_ratio。
  4. 断点: load_state_dict 后 _ohem_step_cpu 恢复 (warmup 进度不丢)。
  5. bf16: in-place 权重乘法不报 dtype 错误, 数值在 bf16 容差内。
  6. stats: last_stats 四元组与 batch/kept 语义正确。
"""
import math
import os
import sys

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29517')
import torch
import torch.nn.functional as F

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group('gloo', rank=0, world_size=1)

import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 按路径直载, 绕过 classifiers/__init__ 的仓库级依赖 (general_utils)
_load('losses.margin_loss', 'losses/margin_loss.py')
_load('losses.adaface', 'losses/adaface.py')
pfc = _load('partial_fc_mod', 'classifiers/partial_fc/partial_fc.py')
DistCrossEntropy = pfc.DistCrossEntropy

torch.manual_seed(2048)
B, C = 64, 100
labels_1d = torch.randint(0, C, (B,))
labels = labels_1d.view(-1, 1)  # PFC 传给 CE 的形状是 (B,1)
base = torch.randn(B, C) * 3.0

# ---------- 1. ratio=1.0 与原实现等价 ----------
ref = F.cross_entropy(base, labels_1d)  # 先算参考: CE 会原地改写传入的 logits
logits = base.clone().requires_grad_(True)
ce = DistCrossEntropy()  # 默认全部关闭
loss = ce(logits, labels)
assert torch.allclose(loss.detach(), ref, atol=1e-5), f"loss mismatch: {loss.item()} vs {ref.item()}"
loss.backward()
lg = base.clone().requires_grad_(True)
F.cross_entropy(lg, labels_1d).backward()
assert torch.allclose(logits.grad, lg.grad, atol=1e-6), "grad mismatch at ratio=1.0"
print("[1] ratio=1.0: loss/grad 与标准 CE 一致  PASS")

# ---------- 2. ratio<1.0 loss/grad 与手写参考一致 ----------
ratio = 0.5
k_expect = math.ceil(ratio * B)
logits2 = base.clone().requires_grad_(True)
ce2 = DistCrossEntropy(ohem_ratio=ratio, ohem_min_kept=8, ohem_warmup_steps=0)
loss2 = ce2(logits2, labels)

# autograd 参考: 全 softmax -> per-sample nll -> top-k 均值
lg2 = base.clone().requires_grad_(True)
p = F.softmax(lg2, dim=1)
nll_ref = -torch.log(p.gather(1, labels.view(-1, 1)).clamp_min(1e-30))
kept_idx = torch.topk(nll_ref.flatten(), k=k_expect, sorted=False).indices
sel = torch.zeros(B, dtype=torch.bool); sel[kept_idx] = True
ref_loss = nll_ref.flatten()[sel].mean()
assert torch.allclose(loss2.detach(), ref_loss.detach(), atol=1e-5), \
    f"ohem loss mismatch: {loss2.item()} vs {ref_loss.item()}"
loss2.backward()
ref_loss.backward()
g_diff = (logits2.grad - lg2.grad).abs().max().item()
assert g_diff < 1e-5, f"ohem grad mismatch, max diff {g_diff}"
# 未选中行梯度为 0
assert logits2.grad[~sel].abs().max().item() == 0.0, "dropped rows should have zero grad"
print(f"[2] ratio={ratio}: loss==top{k_expect}均值, 梯度一致, 未选中行零梯度  PASS")

# ---------- 3. warmup 退火 ----------
ce3 = DistCrossEntropy(ohem_ratio=0.5, ohem_warmup_steps=10)
xs = torch.randn(B, C)
ls = torch.randint(0, C, (B,)).view(-1, 1)
ratios = []
for _ in range(12):
    ce3(xs, ls)
    ratios.append(ce3.last_ratio_eff)
assert ratios[0] == 1.0 and ratios[9] == 1.0 + (0.5 - 1.0) * 9 / 10 and ratios[10] == 0.5 and ratios[11] == 0.5, ratios
print(f"[3] warmup: ratio_eff {ratios[0]:.2f} -> {ratios[10]:.2f} 并保持  PASS")

# ---------- 4. 断点恢复 warmup 计数 ----------
ce4 = DistCrossEntropy(ohem_ratio=0.5, ohem_warmup_steps=10)
for _ in range(7):
    ce4(xs, ls)
sd = ce4.state_dict()
assert 'ohem_step' in sd and int(sd['ohem_step'].item()) == 7
ce5 = DistCrossEntropy(ohem_ratio=0.5, ohem_warmup_steps=10)
missing, unexpected = ce5.load_state_dict(sd, strict=False)
assert ce5._ohem_step_cpu == 7, ce5._ohem_step_cpu
ce5(xs, ls)
assert abs(ce5.last_ratio_eff - (1.0 + (0.5 - 1.0) * 7 / 10)) < 1e-9
# 旧 checkpoint (无 ohem_step) 加载不报错
ce6 = DistCrossEntropy(ohem_ratio=0.5)
ce6.load_state_dict({}, strict=False)
print("[4] 断点: ohem_step 随 state_dict 存取, 旧 ckpt 兼容  PASS")

# ---------- 5. bf16 ----------
logits_bf = base.clone().bfloat16().requires_grad_(True)
ce_bf = DistCrossEntropy(ohem_ratio=0.5, ohem_warmup_steps=0)
loss_bf = ce_bf(logits_bf, labels)
loss_bf.backward()  # 不应出现 in-place mul_ 的 dtype 提升错误
assert torch.isfinite(logits_bf.grad).all()
ref_bf = nll_ref.flatten()[sel].mean()
assert abs(loss_bf.item() - ref_bf.item()) < 0.1, (loss_bf.item(), ref_bf.item())
print(f"[5] bf16: 无 dtype 报错, loss={loss_bf.item():.4f} (ref={ref_bf.item():.4f})  PASS")

# ---------- 6. stats 语义 ----------
s = ce2.last_stats
assert s[2].item() == k_expect and s[3].item() == B
assert abs(s[0].item() - nll_ref.mean().item()) < 1e-4   # 全批均值
assert abs(s[1].item() - ref_loss.item()) < 1e-4         # 选中均值
print(f"[6] stats: kept={int(s[2].item())}/{int(s[3].item())}, loss_all={s[0].item():.4f}, loss_kept={s[1].item():.4f}  PASS")

# ---------- 7. min_kept 下限 ----------
ce7 = DistCrossEntropy(ohem_ratio=0.01, ohem_min_kept=8, ohem_warmup_steps=0)
ce7(xs, ls)
assert ce7.last_stats[2].item() == 8, ce7.last_stats[2].item()
print("[7] min_kept=8 下限生效  PASS")

print("\nALL PASS")

"""量测拓扑损失随 batch N 的缩放.

PartialFC 会 all_gather 把 batch 从 128 放大到 128*8=1024,
拓扑损失(CPU 持久同调)因此要在 1024 个点上跑, 而 FC 路径只跑 128.
"""
import sys, time, torch
sys.path.insert(0, '/root/zhaokj/CVLface/cvlface')
sys.path.insert(0, '/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr')
from losses.topology import compute_topological_loss

dev = 'cuda:0'
print(f"{'N':>6} {'耗时/步(ms)':>12} {'相对128':>9}")
print('-' * 30)
base = None
for N in (128, 256, 512, 1024):
    imgs = torch.randn(N, 3, 112, 112, device=dev)
    emb = torch.randn(N, 512, device=dev, requires_grad=True)
    compute_topological_loss(imgs, emb, use_grad=True)  # warmup
    torch.cuda.synchronize()
    t = time.perf_counter()
    reps = 3
    for _ in range(reps):
        compute_topological_loss(imgs, emb, use_grad=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1000
    if base is None:
        base = ms
    print(f"{N:>6} {ms:>12.1f} {ms/base:>8.1f}x")

print()
print("FC 路径     : 拓扑跑 N=128  (无 all_gather)")
print("PartialFC路径: 拓扑跑 N=1024 (all_gather 8 个 rank)")

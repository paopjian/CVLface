"""DDP 权重一致性校验的低显存实现.

general_utils.dist_utils.verify_ddp_weights_equal 对每个参数做 all_gather, 峰值显存是
    参数大小 x world_size x 2   (tensor_list 一份 + torch.stack 又一份)
人脸识别的 FC 分类头在 791509 类 x 512 维时本身就是 1.62 GB, 8 卡要 ~13 GB, 必然 OOM
(实测报错 "Tried to allocate 12.08 GiB").

那个文件被 cvlface 下所有实验共用, 所以不在原地改; 这里放一份 topofr 专用的等价实现,
只有 topofr/train_opt.py 引用它.
"""
import torch
import torch.distributed as dist

from general_utils.dist_utils import get_world_size


def verify_ddp_weights_equal(model: torch.nn.Module, atol: float = 1e-5,
                             chunk_elems: int = 16 * 1024 * 1024) -> None:
    """逐块比对各 rank 的参数是否与 rank0 一致.

    做法: 把 rank0 的分块 broadcast 过来, 和本地分块逐元素相减取最大绝对差, 再对标量
    all_reduce(MAX). 临时显存被 chunk_elems 限死 (默认 16M 元素 = 64 MB), 与参数规模和
    卡数都无关. 比对精度和原版一致 —— 仍然是逐元素的, 不是抽样或哈希.
    """
    if hasattr(model, "module"):
        model = model.module

    world_size = get_world_size()
    if world_size == 1:
        return print("Skipping DDP verification as world_size=1")

    mismatched = []
    for name, param in model.named_parameters():
        flat = param.data.reshape(-1)
        max_diff = torch.zeros((), device=flat.device, dtype=torch.float32)
        for start in range(0, flat.numel(), chunk_elems):
            local_chunk = flat[start:start + chunk_elems]
            ref = local_chunk.clone()      # clone 保证 contiguous, broadcast 后成为 rank0 的值
            dist.broadcast(ref, src=0)
            max_diff = torch.maximum(max_diff, (local_chunk - ref).abs().max().float())
            del ref
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
        if max_diff.item() > atol:
            mismatched.append((name, max_diff.item()))

    if mismatched:
        print("DDP parameter verification failed ❌")
        for name, diff in mismatched[:10]:
            print(f"  {name}: max abs diff {diff:.3e} > atol {atol}")
    else:
        print("Verified DDP parameter correctness ✅ (low-mem)")

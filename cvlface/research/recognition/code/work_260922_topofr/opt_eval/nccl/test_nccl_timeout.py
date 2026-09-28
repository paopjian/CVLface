"""
测试 eval_all_torch_single.py 中 timeout_minutes 通过 DDPStrategy 是否生效。
模拟 fabric run 启动，设置 NCCL 超时 6 秒，rank 0 sleep 15 秒制造超时。

用法 (和 launcher 一致的启动方式):
  CUDA_VISIBLE_DEVICES=0,1 fabric run --strategy=ddp --devices=2 --precision=32-true test_nccl_timeout.py
"""
import os, sys, time, datetime
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy


def main():
    # 模拟 args.timeout_minutes 传入 (这里用 0.1 分钟 = 6 秒)
    timeout_minutes = 0.1

    ddp_strategy = DDPStrategy(timeout=datetime.timedelta(minutes=timeout_minutes))
    fabric = Fabric(
        precision="32-true",
        accelerator="auto",
        strategy=ddp_strategy,
        devices="auto",
    )
    # 多卡由 fabric run 启动时不调用 fabric.launch()

    rank = fabric.local_rank
    print(f"[Rank {rank}] Fabric 初始化完成, NCCL timeout={timeout_minutes*60:.0f}s")

    # 正常 barrier
    fabric.barrier()
    print(f"[Rank {rank}] 第一次 barrier 成功")

    # rank 0 sleep 制造超时
    if rank == 0:
        print(f"[Rank 0] sleep 15 秒，故意不参与下一次 barrier...")
        time.sleep(15)

    print(f"[Rank {rank}] 开始第二次 barrier...")
    start = time.time()
    fabric.barrier()
    print(f"[Rank {rank}] 第二次 barrier 完成 (耗时 {time.time()-start:.1f}s)")


if __name__ == "__main__":
    main()

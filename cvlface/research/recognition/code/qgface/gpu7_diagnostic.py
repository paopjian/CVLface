"""
GPU 7 硬件故障诊断与复现脚本
==============================
目的: 确认 GPU 7 (PCI:0000:d8:00) 的硬件故障，快速复现问题。

测试场景:
  1. single     - GPU 7 单卡基础操作 (张量创建、矩阵乘法、卷积、反向传播、50轮压力)
  2. multi      - 多卡并行操作 (所有 GPU 同时执行计算，对比耗时; GPU间数据传输; 多卡通信模拟)
  3. ddp        - 多卡 DDP 分布式训练模拟 (NCCL all_reduce/all_gather/broadcast + DDP训练/推理 + 大张量通信)
  4. stress     - 单卡纯运算压力测试 (模型训练 + FP16/FP32 matmul + 卷积, 三流并发, 持续N分钟)
  5. vram       - 显存压力测试 (填满24G, pattern写入验证, 随机校验bit-flip, 满载计算, 反复分配释放)
  6. stress_all - 全卡并行压力测试 (所有GPU同时跑stress, 对比吞吐量, 自动检测异常卡)
  all = single + multi

用法:
  python gpu7_diagnostic.py --test single --gpu 7
  python gpu7_diagnostic.py --test multi
  python gpu7_diagnostic.py --test vram --gpu 7
  python gpu7_diagnostic.py --test stress --gpu 7 --duration 30
  python gpu7_diagnostic.py --test stress_all --duration 30
  torchrun --nproc_per_node=8 gpu7_diagnostic.py --test ddp

  # 对比: 排除 GPU 7
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python gpu7_diagnostic.py --test multi

故障背景:
  dmesg: NVRM: Xid (PCI:0000:d8:00): 119, GSP RPC Timeout
  训练时 NCCL ALLREDUCE 超时 (1800s)，nvidia-smi 显示 GPU 7 为 ERR!
"""

import os
import time
import argparse
import signal
from datetime import datetime
from tqdm import tqdm


def print_header(title):
    w = 70
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def print_result(name, passed, detail="", elapsed=None):
    s = "✔️PASS" if passed else "❌FAIL"
    t = f" ({elapsed:.2f}s)" if elapsed is not None else ""
    print(f"  [{s}] {name}{t}" + (f"  {detail}" if detail else ""))


def timeout_handler(signum, frame):
    raise TimeoutError("操作超时!")


# ============================================================
# 测试 1: 单卡基础操作
# 在指定 GPU 上依次执行: 设备信息读取 → 张量创建(1024x1024)
# → 矩阵乘法(4096x4096) → 大规模matmul(2k/4k/8k) → 卷积前向
# → 反向传播 → 50轮持续matmul, 每项有超时保护
# ============================================================
def test_single_gpu(gpu_id, timeout_sec=60):
    import torch
    import torch.nn as nn

    print_header(f"测试 1: GPU {gpu_id} 单卡基础操作")
    device = f"cuda:{gpu_id}"
    results = []

    # 定义测试项: (名称, 执行函数), 失败时 early-return
    def run_test(name, fn, critical=False):
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)
            t0 = time.time()
            detail = fn()
            torch.cuda.synchronize(device)
            elapsed = time.time() - t0
            signal.alarm(0)
            print_result(name, True, detail or "", elapsed)
            results.append((name, True, ""))
            return True
        except Exception as e:
            signal.alarm(0)
            print_result(name, False, str(e))
            results.append((name, False, str(e)))
            return False

    # 1.1 设备信息
    try:
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  设备: {props.name} | 显存: {props.total_memory / 1024**3:.1f}GB | SM: {props.multi_processor_count}")
        results.append(("设备信息", True, ""))
    except Exception as e:
        print_result("设备信息", False, str(e))
        return [(name, False, str(e))]

    tests = [
        ("张量创建 1024x1024",
         lambda: f"range=[{torch.randn(1024,1024,device=device).min():.3f}, {torch.randn(1024,1024,device=device).max():.3f}]",
         True),
        ("矩阵乘法 4096x4096",
         lambda: f"shape={torch.mm(torch.randn(4096,4096,device=device), torch.randn(4096,4096,device=device)).shape}",
         True),
        ("大规模matmul 2k/4k/8k",
         lambda: [torch.mm(torch.randn(s,s,device=device), torch.randn(s,s,device=device)) for s in [2048,4096,8192]] and
                 f"peak={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB",
         False),
        ("卷积前向 batch=32 112x112",
         lambda: f"out={nn.Conv2d(3,64,3,padding=1).to(device)(torch.randn(32,3,112,112,device=device)).shape}",
         False),
        ("反向传播 MLP",
         lambda: (lambda m, x: (m(x).sum().backward(), f"grad_norm={sum(p.grad.norm().item() for p in m.parameters()):.4f}"))(
             nn.Sequential(nn.Linear(512,1024), nn.ReLU(), nn.Linear(1024,512)).to(device),
             torch.randn(256,512,device=device))[1],
         False),
    ]

    for name, fn, critical in tests:
        ok = run_test(name, fn, critical)
        if not ok and critical:
            return results

    # 1.7 持续压力 (用 tqdm)
    try:
        signal.alarm(timeout_sec * 2)
        pbar = tqdm(range(50), desc="  持续压力 50轮matmul", ncols=80)
        t0 = time.time()
        for i in pbar:
            torch.mm(torch.randn(4096,4096,device=device), torch.randn(4096,4096,device=device))
            if i % 10 == 0:
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        signal.alarm(0)
        pbar.close()
        print_result("持续压力 50轮matmul", True, f"avg {elapsed/50*1000:.1f}ms/轮", elapsed)
        results.append(("持续压力", True, ""))
    except Exception as e:
        signal.alarm(0)
        print_result("持续压力", False, str(e))
        results.append(("持续压力", False, str(e)))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return results


# ============================================================
# 测试 2: 多卡并行操作
# 所有可见GPU同时跑matmul对比耗时 → GPU0到各卡数据传输
# → 多卡reduce-broadcast模拟, 输出统计和异常卡警告
# ============================================================
def test_multi_gpu(timeout_sec=120):
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_gpus = torch.cuda.device_count()
    print_header(f"测试 2: 多卡并行 ({num_gpus} GPU)")

    if num_gpus == 0:
        print("  没有可用 GPU!")
        return

    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 2.1 各卡独立 matmul
    def matmul_on_gpu(gpu_id, size=4096, n_iters=20):
        dev = f"cuda:{gpu_id}"
        try:
            torch.cuda.synchronize(dev)
            t0 = time.time()
            for _ in range(n_iters):
                torch.mm(torch.randn(size,size,device=dev), torch.randn(size,size,device=dev))
            torch.cuda.synchronize(dev)
            return gpu_id, time.time()-t0, True, ""
        except Exception as e:
            return gpu_id, 0, False, str(e)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as ex:
            futs = {ex.submit(matmul_on_gpu, i): i for i in range(num_gpus)}
            res = {}
            for f in tqdm(as_completed(futs), total=num_gpus, desc="  各卡matmul 4096x4096x20", ncols=80):
                gid, el, ok, err = f.result()
                res[gid] = (el, ok, err)
        signal.alarm(0)

        times = []
        for i in sorted(res):
            el, ok, err = res[i]
            if ok:
                print_result(f"GPU {i}", True, elapsed=el)
                times.append(el)
            else:
                print_result(f"GPU {i}", False, err)
        if times:
            avg, mx, mn = sum(times)/len(times), max(times), min(times)
            print(f"  统计: avg={avg:.2f}s min={mn:.2f}s max={mx:.2f}s delta={mx-mn:.2f}s")
            if mx > avg * 2:
                slow = [i for i,(t,s,_) in res.items() if s and t > avg*2]
                print(f"  !! GPU {slow} 明显慢于其他卡!")
    except TimeoutError:
        signal.alarm(0)
        print(f"  TIMEOUT ({timeout_sec}s) - 可能有 GPU 卡死!")
        return

    # 2.2 GPU间传输
    for i in range(1, num_gpus):
        try:
            signal.alarm(timeout_sec)
            t0 = time.time()
            x = torch.randn(1024,1024, device="cuda:0")
            y = x.to(f"cuda:{i}")
            torch.cuda.synchronize(i)
            signal.alarm(0)
            print_result(f"GPU 0→{i} 传输", torch.allclose(x.cpu(),y.cpu()), elapsed=time.time()-t0)
        except Exception as e:
            signal.alarm(0)
            print_result(f"GPU 0→{i} 传输", False, str(e))

    for i in range(num_gpus):
        torch.cuda.empty_cache()


# ============================================================
# 测试 3: DDP 分布式训练
# 用 torchrun 启动, NCCL后端, 测试 all_reduce → all_gather
# → broadcast → DDP模型训练50步 → DDP推理100batch
# → 大张量(64/256/512MB) all_reduce 带宽测试
# ============================================================
def test_ddp(timeout_sec=120):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    if rank == 0:
        print_header(f"测试 3: DDP ({world_size} 进程)")
        for i in range(world_size):
            print(f"  Rank {i}: {torch.cuda.get_device_name(i)}")
    dist.barrier()

    # 3.1 all_reduce
    try:
        t0 = time.time()
        t = torch.ones(1024,1024, device=device) * (rank+1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        exp = world_size*(world_size+1)/2
        if rank == 0:
            print_result("NCCL all_reduce", torch.allclose(t, torch.full_like(t,exp)),
                         f"expect={exp} got={t.mean():.1f}", time.time()-t0)
    except Exception as e:
        if rank == 0: print_result("all_reduce", False, str(e))
        dist.destroy_process_group(); return

    # 3.2 all_gather
    try:
        t0 = time.time()
        t = torch.full((512,512), float(rank), device=device)
        g = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(g, t); torch.cuda.synchronize(device)
        ok = all(torch.allclose(g[i], torch.full_like(t, float(i))) for i in range(world_size))
        if rank == 0: print_result("NCCL all_gather", ok, elapsed=time.time()-t0)
    except Exception as e:
        if rank == 0: print_result("all_gather", False, str(e))

    # 3.3 broadcast
    try:
        t0 = time.time()
        t = torch.full((1024,1024), 42.0, device=device) if rank==0 else torch.zeros(1024,1024, device=device)
        dist.broadcast(t, src=0); torch.cuda.synchronize(device)
        if rank == 0: print_result("NCCL broadcast", torch.allclose(t, torch.full_like(t,42.0)), elapsed=time.time()-t0)
    except Exception as e:
        if rank == 0: print_result("broadcast", False, str(e))

    dist.barrier()

    # 3.4 DDP 训练
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(512,2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                     nn.Linear(2048,2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                     nn.Linear(2048,512))
        def forward(self, x): return self.net(x)

    try:
        model = DDP(M().to(device), device_ids=[local_rank])
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.MSELoss()
        n = 50; t0 = time.time()
        pbar = tqdm(range(n), desc=f"  [Rank{rank}] DDP训练", ncols=80, disable=(rank!=0))
        for step in pbar:
            x = torch.randn(128,512,device=device)
            loss = crit(model(x), torch.randn(128,512,device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            if rank==0: pbar.set_postfix(loss=f"{loss.item():.4f}")
        torch.cuda.synchronize(device); elapsed = time.time()-t0
        if rank == 0:
            print_result(f"DDP训练 {n}步", True, f"loss={loss.item():.4f} {n*128/elapsed:.0f}samp/s", elapsed)
    except Exception as e:
        if rank == 0: print_result("DDP训练", False, str(e))

    # 3.5 DDP 推理
    try:
        model.eval(); t0 = time.time()
        with torch.no_grad():
            for _ in tqdm(range(100), desc=f"  [Rank{rank}] DDP推理", ncols=80, disable=(rank!=0)):
                model(torch.randn(256,512,device=device))
        torch.cuda.synchronize(device); elapsed = time.time()-t0
        if rank == 0: print_result("DDP推理 100x256", True, f"{100*256/elapsed:.0f}samp/s", elapsed)
    except Exception as e:
        if rank == 0: print_result("DDP推理", False, str(e))

    # 3.6 大张量通信
    for mb in [64, 256, 512]:
        try:
            numel = mb*1024*1024//4
            t0 = time.time()
            dist.all_reduce(torch.randn(numel, device=device))
            torch.cuda.synchronize(device); elapsed = time.time()-t0
            if rank == 0: print_result(f"all_reduce {mb}MB", True, f"~{mb*2/elapsed/1024:.1f}GB/s", elapsed)
        except Exception as e:
            if rank == 0: print_result(f"all_reduce {mb}MB", False, str(e))

    dist.barrier()
    dist.destroy_process_group()


# ============================================================
# 测试 5: 显存压力测试
# 填满~23.5GB显存 → 5种pattern(全零/全一/负一/大值/小值)逐块写入验证
# → 5轮随机数据写入+算术变换+checksum校验(检测bit-flip)
# → 满载下做matmul → 50轮x20GB反复分配释放
# ============================================================
def test_vram(gpu_id):
    import torch

    print_header(f"测试 5: GPU {gpu_id} 显存压力测试")
    device = f"cuda:{gpu_id}"

    try:
        props = torch.cuda.get_device_properties(gpu_id)
        total_gb = props.total_memory / 1024**3
        print(f"  {props.name} | {total_gb:.1f} GB")
    except Exception as e:
        print_result("设备访问", False, str(e)); return

    # 5.1 最大分配 (~23.5GB, 预留500MB给CUDA context)
    reserve_mb, chunk_mb = 500, 1024
    target_mb = int(total_gb * 1024) - reserve_mb
    tensors = []
    alloc_mb = 0

    try:
        pbar = tqdm(total=target_mb, desc="  分配显存", unit="MB", ncols=80)
        while alloc_mb + chunk_mb <= target_mb:
            tensors.append(torch.empty(chunk_mb*1024*1024//4, dtype=torch.float32, device=device))
            alloc_mb += chunk_mb
            pbar.update(chunk_mb)
        rem = target_mb - alloc_mb
        if rem > 64:
            tensors.append(torch.empty(rem*1024*1024//4, dtype=torch.float32, device=device))
            alloc_mb += rem
            pbar.update(rem)
        pbar.close()
        print_result(f"分配 {alloc_mb/1024:.1f}GB ({len(tensors)}块)", True,
                     f"占用: {torch.cuda.memory_allocated(device)/1024**3:.2f}/{total_gb:.1f}GB")
    except Exception as e:
        print_result(f"分配 (已{torch.cuda.memory_allocated(device)/1024**3:.1f}GB)", False, str(e))
        if not tensors: return

    # 5.2 Pattern 写入验证 (5种pattern, 写入后逐块校验每个元素)
    patterns = [("全零", 0.0), ("全一", 1.0), ("负一", -1.0), ("大值1e4", 1e4), ("小值1e-6", 1e-6)]
    for pat_name, val in tqdm(patterns, desc="  Pattern验证", ncols=80):
        t0 = time.time()
        try:
            for t in tensors: t.fill_(val)
            torch.cuda.synchronize(device)
            # 用 min/max 校验，避免创建大型 bool 临时张量导致 OOM
            errs = sum(1 for t in tensors if t.min().item() != val or t.max().item() != val)
            torch.cuda.synchronize(device)
            elapsed = time.time() - t0
            bw = alloc_mb * 2 / elapsed / 1024
            print_result(f"Pattern {pat_name}", errs == 0,
                         f"err_chunks={errs}/{len(tensors)}" if errs else f"~{bw:.1f}GB/s", elapsed)
        except Exception as e:
            print_result(f"Pattern {pat_name}", False, str(e))

    # 5.3 随机校验 (写入随机数→算术变换恢复→比对checksum, 检测bit-flip)
    for r in tqdm(range(5), desc="  随机校验", ncols=80):
        t0 = time.time()
        try:
            torch.manual_seed(42+r); torch.cuda.manual_seed(42+r)
            sums_w = []
            for t in tensors:
                t.normal_(0,1); sums_w.append(t.sum().item())
            torch.cuda.synchronize(device)
            # (x*2+1)*0.5-0.5 = x, 增加电路负载后恢复原值
            for t in tensors: t.mul_(2.0).add_(1.0).mul_(0.5).sub_(0.5)
            torch.cuda.synchronize(device)
            sums_r = [t.sum().item() for t in tensors]
            max_diff = max(abs(r-w)/abs(w) if w else abs(r-w) for w,r in zip(sums_w,sums_r))
            ok = max_diff < 1e-3
            print_result(f"随机校验 {r+1}/5", ok, f"max_diff={max_diff:.2e}", time.time()-t0)
        except Exception as e:
            print_result(f"随机校验 {r+1}", False, str(e))

    # 清理循环变量对 tensors 末尾元素的残留引用，否则 del tensors[-1] 无法真正释放显存
    t = None  # noqa: F841 — 5.2/5.3 for-loop 泄露的引用

    # 5.4 满载计算 (保持~22GB占用, 释放1块后做matmul)
    del tensors[-1]; torch.cuda.empty_cache()
    try:
        t0 = time.time()
        for _ in tqdm(range(20), desc="  满载matmul", ncols=80):
            a = torch.randn(2048,2048,device=device)
            c = torch.mm(a, torch.randn(2048,2048,device=device))
            torch.cuda.synchronize(device); del a, c
        elapsed = time.time()-t0
        print_result(f"满载matmul 20轮 ({torch.cuda.memory_allocated(device)/1024**3:.1f}GB占用)",
                     True, f"avg {elapsed/20*1000:.1f}ms", elapsed)
    except Exception as e:
        print_result("满载matmul", False, str(e))

    # 5.5 反复分配释放 (50轮x20GB, 测试显存控制器稳定性)
    del tensors; torch.cuda.empty_cache()
    try:
        t0 = time.time()
        for i in tqdm(range(50), desc="  分配释放 50x20GB", ncols=80):
            ts = [torch.randn(512*1024*1024//4, dtype=torch.float32, device=device)
                  for _ in range(40)]  # 40x512MB = 20GB
            _ = sum(t.sum().item() for t in ts)
            del ts; torch.cuda.empty_cache()
        print_result("反复分配释放 50x20GB", True, f"total=1000GB", time.time()-t0)
    except Exception as e:
        print_result("反复分配释放", False, str(e))

    torch.cuda.empty_cache()


# ============================================================
# 测试 4: 纯运算压力测试 (不占显存, 专注计算单元满载)
# 模型训练(前向+反向+更新) + 并行FP16 matmul(Tensor Core)
# + FP32 matmul + 卷积, 多CUDA流并发, 异步流水线
# 持续N分钟, 出错即停并报告时间点
# 显存压力请用 --test vram 单独测试
# ============================================================
def _stress_worker(gpu_id, duration_min, results_dict=None, silent=False):
    """单卡压力测试工作函数。
    results_dict: 如果提供, 将结果写入 results_dict[gpu_id]
    silent: True 时抑制 tqdm 和大部分输出 (多卡模式下由主线程汇总输出)
    返回 (steps, total_min, errors)
    """
    import torch
    import torch.nn as nn

    device = f"cuda:{gpu_id}"
    torch.backends.cudnn.benchmark = True

    try:
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(gpu_id)
        if not silent:
            print(f"  设备: {props.name} | {props.total_memory/1024**3:.1f}GB | SM: {props.multi_processor_count}")
    except Exception as e:
        res = (0, 0.0, [(0.0, "设备访问", str(e))])
        if results_dict is not None: results_dict[gpu_id] = res
        if not silent: print_result("设备访问", False, str(e))
        return res

    class HeavyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1024,4096), nn.BatchNorm1d(4096), nn.GELU(),
                nn.Linear(4096,4096), nn.BatchNorm1d(4096), nn.GELU(),
                nn.Linear(4096,4096), nn.BatchNorm1d(4096), nn.GELU(),
                nn.Linear(4096,1024))
        def forward(self, x): return self.layers(x)

    try:
        model = HeavyModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        n_params = sum(p.numel() for p in model.parameters())

        batch_size = 512
        x_buf = torch.empty(batch_size, 1024, device=device)
        y_buf = torch.empty(batch_size, 1024, device=device)

        mat_a_f16 = torch.empty(4096, 4096, device=device, dtype=torch.float16)
        mat_b_f16 = torch.empty(4096, 4096, device=device, dtype=torch.float16)
        mat_c_f16 = torch.empty(4096, 4096, device=device, dtype=torch.float16)

        mat_a_f32 = torch.empty(4096, 4096, device=device, dtype=torch.float32)
        mat_b_f32 = torch.empty(4096, 4096, device=device, dtype=torch.float32)
        mat_c_f32 = torch.empty(4096, 4096, device=device, dtype=torch.float32)

        conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
        conv_buf = torch.empty(32, 64, 56, 56, device=device)

        stream_fp16 = torch.cuda.Stream(device=device)
        stream_fp32 = torch.cuda.Stream(device=device)

        if not silent:
            used_gb = torch.cuda.memory_allocated(device) / 1024**3
            total_gb = props.total_memory / 1024**3
            print(f"  模型: {n_params/1e6:.1f}M 参数 | 显存: {used_gb:.1f}/{total_gb:.1f}GB")
            print(f"  负载: 训练(主流) + FP16 matmul(流2) + FP32 matmul(流3) + 卷积")
    except Exception as e:
        res = (0, 0.0, [(0.0, "初始化", str(e))])
        if results_dict is not None: results_dict[gpu_id] = res
        if not silent: print_result("初始化", False, str(e))
        return res

    start = time.time()
    dur_sec = duration_min * 60
    step = 0
    errors = []

    pbar = None
    if not silent:
        print(f"  开始: {datetime.now().strftime('%H:%M:%S')} → "
              f"预计: {datetime.fromtimestamp(start+dur_sec).strftime('%H:%M:%S')}\n")
        pbar = tqdm(total=dur_sec, desc="  压力测试", unit="s", ncols=100, bar_format=
                    '{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}, {postfix}]')
    last_update = start

    try:
        while time.time() - start < dur_sec:
            elapsed_min = (time.time() - start) / 60

            try:
                model.train()
                x_buf.normal_()
                y_buf.normal_()
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = crit(model(x_buf), y_buf)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                conv_buf.normal_()
                _ = conv(conv_buf)
            except Exception as e:
                errors.append((elapsed_min, "训练步", str(e))); break

            try:
                with torch.cuda.stream(stream_fp16):
                    for _ in range(4):
                        mat_a_f16.normal_()
                        mat_b_f16.normal_()
                        torch.mm(mat_a_f16, mat_b_f16, out=mat_c_f16)
            except Exception as e:
                errors.append((elapsed_min, "FP16 matmul", str(e))); break

            try:
                with torch.cuda.stream(stream_fp32):
                    for _ in range(2):
                        mat_a_f32.normal_()
                        mat_b_f32.normal_()
                        torch.mm(mat_a_f32, mat_b_f32, out=mat_c_f32)
            except Exception as e:
                errors.append((elapsed_min, "FP32 matmul", str(e))); break

            step += 1
            now = time.time()
            if now - last_update >= 2.0:
                torch.cuda.synchronize(device)
                if pbar:
                    pbar.n = min(now - start, dur_sec)
                    mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                    pbar.set_postfix_str(f"step={step} loss={loss.item():.3f} mem={mem_gb:.1f}GB")
                    pbar.refresh()
                last_update = now

    except KeyboardInterrupt:
        pass

    if pbar: pbar.close()

    try:
        torch.cuda.synchronize(device)
    except Exception as e:
        errors.append(((time.time()-start)/60, "最终同步", str(e)))

    total_min = (time.time() - start) / 60
    throughput = step / total_min if total_min > 0 else 0

    del mat_a_f16, mat_b_f16, mat_c_f16, mat_a_f32, mat_b_f32, mat_c_f32
    del x_buf, y_buf, conv_buf
    torch.cuda.empty_cache()

    res = (step, total_min, errors, throughput)
    if results_dict is not None:
        results_dict[gpu_id] = res
    return res


def test_stress(gpu_id, duration_min=30):
    """单卡纯运算压力测试"""
    print_header(f"测试 4: GPU {gpu_id} 纯运算压力测试 ({duration_min}min)")
    step, total_min, errors, throughput = _stress_worker(gpu_id, duration_min)

    print(f"\n  运行: {total_min:.1f}min, {step}步, {throughput:.0f}步/min")
    if errors:
        print(f"  !! {len(errors)} 个错误:")
        for t, phase, err in errors:
            print(f"     [{t:.1f}min] {phase}: {err}")
        print(f"\n  结论: GPU {gpu_id} 在 {errors[0][0]:.1f}min 后故障!")
        print(f"  检查: dmesg | grep -i xid | tail -10")
    else:
        print(f"  GPU {gpu_id} 在 {total_min:.1f}min 纯运算压力下未出错")
        if total_min < duration_min:
            print(f"  (未跑满 {duration_min}min)")


# ============================================================
# 测试 6: 多卡并行压力测试
# 所有可见 GPU 同时跑纯运算压力 (每卡一个线程),
# 实时汇总进度, 对比吞吐量, 识别异常卡
# ============================================================
def test_stress_all(duration_min=30):
    import torch
    from concurrent.futures import ThreadPoolExecutor

    num_gpus = torch.cuda.device_count()
    print_header(f"测试 6: 全卡并行压力测试 ({num_gpus} GPU × {duration_min}min)")

    if num_gpus == 0:
        print("  没有可用 GPU!")
        return

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} | {props.total_memory/1024**3:.1f}GB | SM: {props.multi_processor_count}")
    print(f"\n  每卡负载: 模型训练 + FP16 matmul×4 + FP32 matmul×2 + 卷积 (三流并发)")

    start = time.time()
    dur_sec = duration_min * 60
    results_dict = {}

    print(f"\n  开始: {datetime.now().strftime('%H:%M:%S')} → "
          f"预计: {datetime.fromtimestamp(start+dur_sec).strftime('%H:%M:%S')}  (Ctrl+C中止)")

    # 主进度条 (只在主线程显示)
    pbar = tqdm(total=dur_sec, desc="  全卡压力", unit="s", ncols=100, bar_format=
                '{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}, {postfix}]')

    # 启动所有卡的工作线程
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for gpu_id in range(num_gpus):
            f = executor.submit(_stress_worker, gpu_id, duration_min, results_dict, silent=True)
            futures[f] = gpu_id

        # 主线程更新进度条
        try:
            while not all(f.done() for f in futures):
                time.sleep(2.0)
                elapsed = time.time() - start
                pbar.n = min(elapsed, dur_sec)
                done_count = sum(1 for f in futures if f.done())
                pbar.set_postfix_str(f"运行中: {num_gpus - done_count}/{num_gpus}卡")
                pbar.refresh()
        except KeyboardInterrupt:
            pass

    pbar.n = min(time.time() - start, dur_sec)
    pbar.refresh()
    pbar.close()

    # ---- 汇总结果 ----
    print_header(f"全卡压力测试结果 ({num_gpus} GPU × {duration_min}min)")

    all_ok = True
    throughputs = {}
    for gpu_id in sorted(results_dict.keys()):
        step, total_min, errors, throughput = results_dict[gpu_id]
        throughputs[gpu_id] = throughput
        if errors:
            all_ok = False
            err_time = errors[0][0]
            err_phase = errors[0][1]
            err_msg = errors[0][2][:80]
            print_result(f"GPU {gpu_id}", False,
                         f"{step}步 {total_min:.1f}min → {err_time:.1f}min [{err_phase}] {err_msg}")
        else:
            print_result(f"GPU {gpu_id}", True,
                         f"{step}步 {total_min:.1f}min {throughput:.0f}步/min",
                         elapsed=total_min * 60)

    # 吞吐量统计与异常检测
    if throughputs:
        vals = list(throughputs.values())
        avg_tp = sum(vals) / len(vals)
        max_tp = max(vals)
        min_tp = min(vals)
        print(f"\n  吞吐量统计: avg={avg_tp:.0f} min={min_tp:.0f} max={max_tp:.0f} 步/min")

        # 检测明显慢的卡 (低于平均的 70%)
        slow_gpus = [gid for gid, tp in throughputs.items() if tp < avg_tp * 0.7]
        if slow_gpus:
            print(f"  ⚠️  GPU {slow_gpus} 吞吐量明显低于平均 (<70%), 可能有硬件问题!")

        # 检测出错的卡
        fail_gpus = [gid for gid in sorted(results_dict.keys())
                     if results_dict[gid][2]]  # errors not empty
        if fail_gpus:
            print(f"  ❌ GPU {fail_gpus} 在压力测试中出错!")
            print(f"  检查: dmesg | grep -i xid | tail -20")

    if all_ok:
        total_steps = sum(r[0] for r in results_dict.values())
        print(f"\n  ✅ 全部 {num_gpus} 卡在 {duration_min}min 压力下未出错, 共 {total_steps} 步")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 硬件故障诊断")
    parser.add_argument("--test", type=str, default="all",
                        choices=["single", "multi", "ddp", "all", "stress", "stress_all", "vram"],
                        help="single/multi/ddp/all/stress/stress_all/vram")
    parser.add_argument("--gpu", type=int, default=7, help="目标GPU (默认7)")
    parser.add_argument("--timeout", type=int, default=60, help="单项超时秒数")
    parser.add_argument("--duration", type=int, default=30, help="stress持续分钟数")
    args = parser.parse_args()

    # ---- 单卡测试: 限制 CUDA 只看到目标卡, 避免在 GPU 0 上白占 ~400MB context ----
    # CUDA 运行时初始化时默认在 GPU 0 建 primary context, 即使你只用 GPU 7。
    # 通过 CUDA_VISIBLE_DEVICES 让目标卡成为唯一可见设备 (映射为 cuda:0),
    # 这样整个进程只在一张卡上分配显存。
    # 对 multi/ddp/all 不能限制, 因为它们需要所有卡。
    single_gpu_tests = ("single", "stress", "vram")
    original_gpu_id = args.gpu
    if args.test in single_gpu_tests and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.gpu = 0  # 限制后目标卡变成 cuda:0
        print(f"  [自动设置 CUDA_VISIBLE_DEVICES={original_gpu_id}, 目标卡映射为 cuda:0]")

    print(f"{'#'*70}")
    print(f"#  GPU 硬件故障诊断  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {os.uname().nodename}")
    print(f"{'#'*70}")

    import torch
    print(f"  PyTorch={torch.__version__}  CUDA={torch.version.cuda}  "
          f"cuDNN={torch.backends.cudnn.version()}  GPU数={torch.cuda.device_count()}"
          + (f"  (物理GPU {original_gpu_id})" if args.test in single_gpu_tests else ""))

    if args.test == "stress":
        test_stress(args.gpu, duration_min=args.duration)
    elif args.test == "stress_all":
        test_stress_all(duration_min=args.duration)
    elif args.test == "vram":
        test_vram(args.gpu)
    elif args.test == "ddp":
        test_ddp(timeout_sec=args.timeout)
    else:
        if args.test in ("single", "all"):
            results = test_single_gpu(args.gpu, timeout_sec=args.timeout)
            p = sum(1 for _,s,_ in results if s)
            print(f"\n  单卡: {p}/{len(results)} 通过" + (f" !! GPU {args.gpu} 异常!" if p<len(results) else ""))
        if args.test in ("multi", "all"):
            test_multi_gpu(timeout_sec=args.timeout)

        print_header("总结")
        print("  失败/超时 → 硬件故障确认")
        print("  短期: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6")
        print("  长期: 联系供应商更换 GPU 7")
        print(f"  DDP: torchrun --nproc_per_node=8 gpu7_diagnostic.py --test ddp")

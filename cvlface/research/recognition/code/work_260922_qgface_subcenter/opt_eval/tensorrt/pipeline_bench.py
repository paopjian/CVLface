"""主进程链路流水线优化: pin_memory + 双缓冲双 stream (H2D 与 engine 重叠)

decode_e2e_bench 定位: 零解码直通也只有 3,108 img/s → 瓶颈在主进程串行链
(collate → H2D → flip → engine → sync), 解码已被 nw 并行掩盖.

模式:
  A sync_baseline : 生产式 (pinned batch, 同 stream 同步 H2D + engine, 每 batch sync)
  B pipeline      : 双缓冲 + copy_stream/compute_stream 重叠

数据: 预生成张量池 (零解码), 隔离解码变量; DataLoader worker 保留真实 collate/pin 路径
"""
import os
import sys
import time

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
ENGINE_BATCH = 256
N_LIMIT = 120000
NW = 8


class PreDecodedDataset(torch.utils.data.Dataset):
    def __init__(self, n, pool=4096):
        g = torch.Generator().manual_seed(0)
        self.pool = torch.rand(pool, 3, 112, 112, generator=g) * 2 - 1
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.pool[idx % len(self.pool)]


def _collate(batch):
    return torch.stack(batch)


def load_engine():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(ENGINE, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
    ctx = engine.create_execution_context()
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_name = n
        else:
            out_name = n
    return engine, ctx, in_name, out_name


def bench_sync(engine, ctx, in_name, out_name, loader):
    """A: 生产式同步 (每 batch: H2D → execute → sync, 单一 buffer)"""
    import tensorrt as trt
    _trt2torch = {trt.float16: torch.float16, trt.float32: torch.float32}
    in_dtype = _trt2torch[engine.get_tensor_dtype(in_name)]
    d_in = torch.zeros(ENGINE_BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
    d_out = torch.zeros(ENGINE_BATCH, 512, dtype=torch.float32, device='cuda')
    ctx.set_tensor_address(in_name, d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    stream = torch.cuda.current_stream()
    t0 = time.time()
    n = 0
    for x in loader:
        x = x.cuda()  # pinned → 同步 H2D
        x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        b2 = x_combined.shape[0]
        # 静态 engine: 尾部批部分拷贝补齐, 输出取前 b2 行
        d_in[:b2].copy_(x_combined.to(in_dtype))
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        n += x.shape[0]
    return n / (time.time() - t0)


def bench_pipeline(engine, ctx, in_name, out_name, loader):
    """B: 双缓冲双 stream, H2D(k+1) 与 execute(k) 重叠"""
    import tensorrt as trt
    _trt2torch = {trt.float16: torch.float16, trt.float32: torch.float32}
    in_dtype = _trt2torch[engine.get_tensor_dtype(in_name)]
    NB = 2
    d_in = [torch.zeros(ENGINE_BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
            for _ in range(NB)]
    d_out = [torch.zeros(ENGINE_BATCH, 512, dtype=torch.float32, device='cuda')
             for _ in range(NB)]
    copy_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()
    ev_copy = [torch.cuda.Event(), torch.cuda.Event()]
    ev_comp = [torch.cuda.Event(), torch.cuda.Event()]

    t0 = time.time()
    n = 0
    it = iter(loader)
    prev = None
    k = 0
    while True:
        try:
            x = next(it)
        except StopIteration:
            break
        buf = k % NB
        # 上一轮 compute 完成才可复用其输出/输入 buffer
        if prev is not None:
            ev_comp[prev].synchronize()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(ev_comp[buf])  # 该 buffer 上一轮 compute 已完
            x_gpu = x.cuda(non_blocking=True)
            x_combined = torch.cat([x_gpu, torch.flip(x_gpu, dims=[3])], dim=0)
            b2 = x_combined.shape[0]
            d_in[buf][:b2].copy_(x_combined.to(in_dtype), non_blocking=True)
            ev_copy[buf].record(copy_stream)
        with torch.cuda.stream(comp_stream):
            comp_stream.wait_event(ev_copy[buf])
            ctx.set_tensor_address(in_name, d_in[buf].data_ptr())
            ctx.set_tensor_address(out_name, d_out[buf].data_ptr())
            ctx.execute_async_v3(comp_stream.cuda_stream)
            ev_comp[buf].record(comp_stream)
        prev = buf
        n += x.shape[0]
        k += 1
    ev_comp[prev].synchronize()
    return n / (time.time() - t0)


def main():
    torch.cuda.set_device(0)
    engine, ctx, in_name, out_name = load_engine()
    subset = PreDecodedDataset(N_LIMIT)
    rows = []
    for pin, mode in [(False, 'baseline_nopin'), (True, 'baseline_pin')]:
        loader = torch.utils.data.DataLoader(
            subset, batch_size=ENGINE_BATCH // 2, num_workers=NW,
            collate_fn=_collate, pin_memory=pin, persistent_workers=True)
        for _ in range(2):  # 预热
            for x in loader:
                break
        thr = bench_sync(engine, ctx, in_name, out_name, loader)
        print(f"{mode}: {thr:,.0f} img/s")
        rows.append({'mode': mode, 'img_per_sec': round(thr, 0)})
        del loader
    loader = torch.utils.data.DataLoader(
        subset, batch_size=ENGINE_BATCH // 2, num_workers=NW,
        collate_fn=_collate, pin_memory=True, persistent_workers=True,
        prefetch_factor=4)
    for _ in range(2):
        for x in loader:
            break
    thr = bench_pipeline(engine, ctx, in_name, out_name, loader)
    print(f"pipeline_double_buffer: {thr:,.0f} img/s")
    rows.append({'mode': 'pipeline_double_buffer', 'img_per_sec': round(thr, 0)})

    import polars as pl
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'pipeline_results.parquet')
    pl.DataFrame(rows).write_parquet(out)
    print(f"结果已保存: {out}")


if __name__ == '__main__':
    main()

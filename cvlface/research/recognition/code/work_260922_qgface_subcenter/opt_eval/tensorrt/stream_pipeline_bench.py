"""FP16 TRT engine 执行模式优化对比: default stream / 专用 stream / CUDA Graph / 双缓冲流水线

背景: 生产 TRTInfer 在 default stream 上 execute_async_v3 + 每 batch synchronize,
TRT 会额外插入 cudaStreamSynchronize 造成开销. 本脚本在纯 engine 层面隔离测量:
  A default_stream  : 复刻生产 (基线)
  B side_stream     : 专用 stream, 消除 TRT 内部 sync 附加开销
  C cuda_graph      : 图捕获整个 execute, 消除 launch 开销 (输入常驻 GPU)
  D double_buffer   : 双 stream 流水线, H2D(k+1) 与 execute(k) 重叠 (模拟真实管线)

用法: python stream_pipeline_bench.py [engine_path]
"""
import os
import sys
import time

import numpy as np
import torch

BATCH = 256
WARMUP = 10
ITERS = 100
ENGINE = sys.argv[1] if len(sys.argv) > 1 else '/tmp/int8_ptq_work/engine_fp16.plan'


def load_runner():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(ENGINE, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
    context = engine.create_execution_context()
    input_name = output_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            input_name = n
        else:
            output_name = n
    _trt2torch = {trt.float16: torch.float16, trt.float32: torch.float32}
    in_dtype = _trt2torch[engine.get_tensor_dtype(input_name)]
    context.set_input_shape(input_name, (BATCH, 3, 112, 112))
    return engine, context, input_name, output_name, in_dtype


def bench_default_stream(engine, context, input_name, output_name, in_dtype):
    """A: 复刻生产 TRTInfer (default stream + 每 batch sync)"""
    d_in = torch.zeros(BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
    d_out = torch.zeros(BATCH, 512, dtype=torch.float32, device='cuda')
    context.set_tensor_address(input_name, d_in.data_ptr())
    context.set_tensor_address(output_name, d_out.data_ptr())
    stream = torch.cuda.current_stream()

    def step():
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        step()
    return ITERS * BATCH / (time.time() - t0)


def bench_side_stream(engine, context, input_name, output_name, in_dtype):
    """B: 专用非默认 stream"""
    d_in = torch.zeros(BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
    d_out = torch.zeros(BATCH, 512, dtype=torch.float32, device='cuda')
    context.set_tensor_address(input_name, d_in.data_ptr())
    context.set_tensor_address(output_name, d_out.data_ptr())
    stream = torch.cuda.Stream()

    def step():
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    for _ in range(WARMUP):
        step()
    stream.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        step()
    stream.synchronize()
    return ITERS * BATCH / (time.time() - t0)


def bench_cuda_graph(engine, context, input_name, output_name, in_dtype):
    """C: CUDA Graph 捕获 execute (静态 shape, 输入常驻固定 buffer)"""
    d_in = torch.zeros(BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
    d_out = torch.zeros(BATCH, 512, dtype=torch.float32, device='cuda')
    context.set_tensor_address(input_name, d_in.data_ptr())
    context.set_tensor_address(output_name, d_out.data_ptr())
    stream = torch.cuda.Stream()

    # 预热 (TRT context 首次执行不能在 capture 内)
    for _ in range(3):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    x = torch.randn(BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
    for _ in range(WARMUP):
        d_in.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        graph.replay()
    torch.cuda.synchronize()
    return ITERS * BATCH / (time.time() - t0)


def bench_double_buffer(engine, context, input_name, output_name, in_dtype):
    """D: 双 stream 流水线, H2D(k+1) 与 execute(k) 重叠 (模拟真实管线输入)"""
    host = [torch.randn(BATCH, 3, 112, 112).pin_memory().to(in_dtype)
            for _ in range(2)]
    d_in = [torch.zeros(BATCH, 3, 112, 112, dtype=in_dtype, device='cuda')
            for _ in range(2)]
    d_out = [torch.zeros(BATCH, 512, dtype=torch.float32, device='cuda')
             for _ in range(2)]
    copy_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()
    ev_copy = [torch.cuda.Event(), torch.cuda.Event()]
    ev_comp = [torch.cuda.Event(), torch.cuda.Event()]

    def step(k):
        i = k % 2
        # copy: 等 buffer i 上一轮 compute 完成 (避免覆盖正在使用的输入)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(ev_comp[i])
            d_in[i].copy_(host[i], non_blocking=True)
            ev_copy[i].record(copy_stream)
        # compute: 等本轮 copy 完成
        with torch.cuda.stream(comp_stream):
            comp_stream.wait_event(ev_copy[i])
            context.set_tensor_address(input_name, d_in[i].data_ptr())
            context.set_tensor_address(output_name, d_out[i].data_ptr())
            context.execute_async_v3(comp_stream.cuda_stream)
            ev_comp[i].record(comp_stream)

    for k in range(WARMUP):
        step(k)
    comp_stream.synchronize()
    t0 = time.time()
    for k in range(ITERS):
        step(k)
    comp_stream.synchronize()
    return ITERS * BATCH / (time.time() - t0)


def main():
    torch.cuda.set_device(0)
    parts = load_runner()
    x = torch.randn(BATCH, 3, 112, 112, dtype=parts[4], device='cuda')

    rows = []
    for name, fn in [
        ('default_stream', bench_default_stream),
        ('side_stream', bench_side_stream),
        ('cuda_graph', bench_cuda_graph),
        ('double_buffer', bench_double_buffer),
    ]:
        try:
            thr = fn(*parts)
        except Exception as e:
            print(f"{name}: 失败 {type(e).__name__}: {e}")
            continue
        print(f"{name}: {thr:,.0f} img/s")
        rows.append({'mode': name, 'img_per_sec': round(thr, 0)})

    base = rows[0]['img_per_sec'] if rows else 1
    for r in rows:
        r['speedup_vs_default'] = round(r['img_per_sec'] / base, 3)
        print(f"  {r['mode']}: {r['img_per_sec']:,.0f} img/s "
              f"({r['speedup_vs_default']:.2f}x)")

    import polars as pl
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'stream_pipeline_results.parquet')
    pl.DataFrame(rows).write_parquet(out)
    print(f"结果已保存: {out}")


if __name__ == '__main__':
    main()

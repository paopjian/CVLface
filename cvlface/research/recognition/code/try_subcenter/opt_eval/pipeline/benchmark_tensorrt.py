"""
TensorRT vs torch.compile 推理加速对比 benchmark
单卡测试, 10000 张随机图, bs=256, iResNet-101
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os, sys, time
sys.path.append(os.path.join(root))

import numpy as np
np.bool = np.bool_

import torch
torch.set_float32_matmul_precision('high')

from models import get_model
from general_utils.config_utils import load_config


def load_model():
    ckpt_path = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'
    model_config = load_config(os.path.join(ckpt_path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(ckpt_path, 'model.pt'))
    model.eval()
    model.cuda()
    return model


def benchmark_method(name, forward_fn, dummy_input, n_images=10000, batch_size=256, warmup=5):
    """通用 benchmark 函数"""
    n_batches = n_images // batch_size

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = forward_fn(dummy_input)
    torch.cuda.synchronize()

    # benchmark
    start = time.time()
    for _ in range(n_batches):
        with torch.no_grad():
            _ = forward_fn(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = n_images / elapsed
    print(f"  {name}: {elapsed:.2f}s, {throughput:.0f} img/s")
    return elapsed, throughput


def export_onnx(model, onnx_path, batch_size=256, fp16=False):
    """导出 ONNX 模型 (fp16=True 时导出半精度, TRT 11 需要此方式启用 FP16)"""
    if fp16:
        export_model = model.half()
        dummy = torch.randn(batch_size, 3, 112, 112, device='cuda', dtype=torch.float16)
    else:
        export_model = model
        dummy = torch.randn(batch_size, 3, 112, 112, device='cuda')
    with torch.no_grad():
        torch.onnx.export(
            export_model, dummy, onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,  # 固定 batch size
            opset_version=17,
            dynamo=False,  # 使用 TorchScript exporter 确保包含权重
        )
    print(f"  ONNX 导出完成: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")


def build_trt_engine(onnx_path, engine_path):
    """从 ONNX 构建 TensorRT engine (TRT 11 API, 精度由 ONNX 模型决定)"""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()  # TRT 11: explicit batch is default
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX Parse Error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    # TRT 11: FP16/INT8 flags removed; precision determined by ONNX model dtype

    print(f"  构建 TensorRT engine...")
    build_start = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - build_start
    print(f"  Engine 构建耗时: {build_time:.2f}s")

    if serialized_engine is None:
        print("  Engine 构建失败!")
        return None

    print(f"  Engine size: {serialized_engine.nbytes/1024/1024:.1f} MB")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"  Engine 保存到: {engine_path}")

    return serialized_engine


def benchmark_trt(engine_bytes, batch_size=256, n_images=10000, warmup=5, fp16=False):
    """TensorRT 推理 benchmark"""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(memoryview(engine_bytes))
    context = engine.create_execution_context()

    # 分配 GPU buffers
    input_shape = (batch_size, 3, 112, 112)
    output_shape = (batch_size, 512)
    dtype = torch.float16 if fp16 else torch.float32

    d_input = torch.randn(input_shape, dtype=dtype, device='cuda')
    d_output = torch.empty(output_shape, dtype=dtype, device='cuda')

    # 设置 tensor addresses
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())

    stream = torch.cuda.Stream()
    n_batches = n_images // batch_size

    # warmup
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # benchmark (不在循环内重新生成数据, 公平对比纯推理吞吐)
    start = time.time()
    for _ in range(n_batches):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    elapsed = time.time() - start

    throughput = n_images / elapsed
    label = "TensorRT FP16" if fp16 else "TensorRT FP32"
    print(f"  {label}: {elapsed:.3f}s, {throughput:.0f} img/s")
    return elapsed, throughput


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    batch_size = 256
    n_images = 10000
    dummy_input = torch.randn(batch_size, 3, 112, 112, device='cuda')

    print("=" * 60)
    print("推理加速技术对比 (iResNet-101, 10000张, bs=256, 单卡)")
    print("=" * 60)

    # 加载模型
    print("\n加载模型...")
    model = load_model()

    results = {}

    # 1. Eager FP32
    print("\n--- 方法1: Eager FP32 ---")
    _, tp = benchmark_method("Eager FP32", model, dummy_input, n_images, batch_size)
    results['Eager FP32'] = tp

    # 2. AMP bf16
    print("\n--- 方法2: AMP bf16 ---")
    def amp_bf16_forward(x):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return model(x)
    _, tp = benchmark_method("AMP bf16", amp_bf16_forward, dummy_input, n_images, batch_size)
    results['AMP bf16'] = tp

    # 3. AMP fp16
    print("\n--- 方法3: AMP fp16 ---")
    def amp_fp16_forward(x):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            return model(x)
    _, tp = benchmark_method("AMP fp16", amp_fp16_forward, dummy_input, n_images, batch_size)
    results['AMP fp16'] = tp

    # 4. torch.compile (reduce-overhead) + bf16
    print("\n--- 方法4: torch.compile (reduce-overhead) + bf16 ---")
    compiled_model_ro = torch.compile(model, mode="reduce-overhead")
    def compiled_ro_forward(x):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return compiled_model_ro(x)
    _, tp = benchmark_method("compile reduce-overhead", compiled_ro_forward, dummy_input, n_images, batch_size)
    results['compile reduce-overhead + bf16'] = tp

    # 5. torch.compile (max-autotune) + bf16
    print("\n--- 方法5: torch.compile (max-autotune) + bf16 ---")
    compiled_model_ma = torch.compile(model, mode="max-autotune")
    def compiled_ma_forward(x):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return compiled_model_ma(x)
    _, tp = benchmark_method("compile max-autotune", compiled_ma_forward, dummy_input, n_images, batch_size)
    results['compile max-autotune + bf16'] = tp

    # 6. TensorRT FP16
    print("\n--- 方法6: TensorRT FP16 ---")
    onnx_path = '/tmp/iresnet101_bs256_fp16.onnx'
    engine_path = '/tmp/iresnet101_bs256_fp16.engine'

    try:
        print("  导出 FP16 ONNX (model.half())...")
        export_onnx(model, onnx_path, batch_size, fp16=True)

        print("  构建 TensorRT Engine...")
        engine_bytes = build_trt_engine(onnx_path, engine_path)

        if engine_bytes:
            _, tp = benchmark_trt(engine_bytes, batch_size, n_images, fp16=True)
            results['TensorRT FP16'] = tp
        else:
            print("  TensorRT 构建失败，跳过")
            results['TensorRT FP16'] = 0
    except Exception as e:
        print(f"  TensorRT 失败: {e}")
        import traceback
        traceback.print_exc()
        results['TensorRT FP16'] = 0

    # 汇总
    print("\n")
    print("=" * 60)
    print("汇总 (10000 张图, bs=256, 单卡 RTX4090)")
    print("=" * 60)

    base_tp = results.get('Eager FP32', 1)
    for name, tp in results.items():
        speedup = tp / base_tp if base_tp > 0 else 0
        print(f"  {name:<35}: {tp:>6.0f} img/s  {speedup:.2f}x")

    print("=" * 60)

    # 清理
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    if os.path.exists(engine_path):
        os.remove(engine_path)

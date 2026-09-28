"""
评估流水线加速方案综合对比
4 种配置在 val_20260605.yaml 上的完整评估 (7卡 DDP):
  1. 无 compile, BF16 (基线)
  2. 预 compile (max-autotune) + BF16
  3. 预 compile (max-autotune) + FP16
  4. TensorRT FP16

记录: 整体时间, compile 生成/加载时间, TRT 制作/加载时间, 各评估器耗时

用法:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run \
    --strategy=ddp --devices=7 --precision=bf16-mixed \
    benchmark_eval_pipeline.py

  单卡测试:
  python benchmark_eval_pipeline.py --num_gpu 1
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os, sys, time, json
sys.path.append(os.path.join(root))

import numpy as np
np.bool = np.bool_

import torch
torch.set_float32_matmul_precision('high')

from models import get_model
from aligners import get_aligner
from evaluations import get_evaluator_by_name
from pipelines import pipeline_from_name
from pipelines.base import BasePipeline
from general_utils.config_utils import load_config
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset


class TRTPipeline(BasePipeline):
    """TensorRT FP16 推理 Pipeline，替换 InferModelPipeline"""

    def __init__(self, engine_path, batch_size=256, model_config=None):
        super().__init__()
        import tensorrt as trt

        self.batch_size = batch_size
        self.color_space = 'RGB'
        self._model_config = model_config

        # 加载 engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(memoryview(engine_bytes))
        self.context = self.engine.create_execution_context()

        # 获取 IO 名称
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # 预分配 buffer
        self.d_input = torch.zeros(batch_size, 3, 112, 112, dtype=torch.float16, device='cuda')
        self.d_output = torch.zeros(batch_size, 512, dtype=torch.float16, device='cuda')
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())
        self.stream = torch.cuda.Stream()

    @property
    def module_names_list(self):
        return []

    def integrity_check(self, dataset_color_space):
        assert dataset_color_space == 'RGB'
        self.color_space = dataset_color_space

    def make_test_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __call__(self, batch):
        x = batch
        actual_bs = x.shape[0]
        x_fp16 = x.half()

        if actual_bs == self.batch_size:
            self.d_input.copy_(x_fp16)
        else:
            # 最后一个 batch 不足，padding
            self.d_input[:actual_bs].copy_(x_fp16)

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        output = self.d_output[:actual_bs].float()
        return output

    def eval(self):
        pass

    def train(self):
        raise NotImplementedError


def build_or_load_trt_engine(model, batch_size=256, cache_dir='/tmp/trt_cache'):
    """构建或加载 TRT FP16 engine，返回 (engine_path, build_time)"""
    import tensorrt as trt

    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, f'iresnet101_bs{batch_size}_fp16.onnx')
    engine_path = os.path.join(cache_dir, f'iresnet101_bs{batch_size}_fp16.engine')

    # 如果 engine 已存在，直接加载
    if os.path.exists(engine_path):
        load_start = time.time()
        # 验证能否加载
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
        if engine is not None:
            load_time = time.time() - load_start
            del engine
            return engine_path, 0, load_time
        # engine 无效，重建
        os.remove(engine_path)

    # 导出 ONNX
    build_start = time.time()
    model_fp16 = model.half().cuda()
    dummy = torch.randn(batch_size, 3, 112, 112, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(
            model_fp16, dummy, onnx_path,
            input_names=['input'], output_names=['output'],
            dynamic_axes=None, opset_version=17, dynamo=False,
        )

    # 构建 engine
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    serialized = builder.build_serialized_network(network, config)
    build_time = time.time() - build_start

    with open(engine_path, 'wb') as f:
        f.write(serialized)

    # 测量加载时间
    load_start = time.time()
    runtime2 = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime2.deserialize_cuda_engine(memoryview(f.read()))
    load_time = time.time() - load_start
    del engine

    # 清理 ONNX
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    return engine_path, build_time, load_time


def timed_extract(evaluator, pipeline):
    """特征提取计时"""
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        collection = evaluator.extract(pipeline)
        collection_flip = evaluator.extract(pipeline, flip_images=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return collection, collection_flip, elapsed


def run_evaluation(fabric, eval_config, eval_pipeline, transform, config_name):
    """运行完整评估，返回各阶段计时"""
    timing = {'config': config_name, 'evaluators': {}}
    total_extract = 0
    total_compute = 0
    total_start = time.time()

    for name, info in eval_config.per_epoch_evaluations.items():
        eval_data_path = os.path.join(eval_config.data_root, info.path)
        eval_type = info.evaluation_type

        if not os.path.isdir(eval_data_path):
            if fabric.local_rank == 0:
                print(f"    [跳过] {name}: 数据不存在")
            continue

        try:
            evaluator = get_evaluator_by_name(
                eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                transform=transform, fabric=fabric,
                batch_size=info.batch_size, num_workers=info.num_workers
            )
            evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        except Exception as e:
            if fabric.local_rank == 0:
                print(f"    [错误] {name}: {e}")
            continue

        # 特征提取
        collection, collection_flip, extract_time = timed_extract(evaluator, eval_pipeline)
        total_extract += extract_time

        # 评估计算 (rank 0)
        compute_time = 0
        if fabric.local_rank == 0:
            compute_start = time.time()
            try:
                evaluator.compute_metric(collection, collection_flip)
            except Exception:
                pass
            compute_time = time.time() - compute_start
            total_compute += compute_time

        fabric.barrier()

        timing['evaluators'][name] = {
            'extract': extract_time, 'compute': compute_time
        }
        if fabric.local_rank == 0:
            print(f"    {name}: extract={extract_time:.1f}s, compute={compute_time:.1f}s")

        del collection, collection_flip
        torch.cuda.empty_cache()

    timing['total_extract'] = total_extract
    timing['total_compute'] = total_compute
    timing['total'] = time.time() - total_start
    return timing


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--skip_trt', action='store_true', help='跳过 TRT 测试')
    args = parser.parse_args()

    ckpt_path = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'
    eval_config_path = 'evaluations/configs/val_20260605.yaml'

    # Fabric 初始化
    csv_logger_dir = os.path.join(root, 'research/recognition/experiments', 'benchmark_pipeline')
    os.makedirs(csv_logger_dir, exist_ok=True)

    fabric = Fabric(
        precision=args.precision,
        accelerator="auto",
        strategy="ddp",
        devices=args.num_gpu,
        loggers=[CSVLogger(root_dir=csv_logger_dir, flush_logs_every_n_steps=1)],
    )
    if args.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=2048)

    # 加载基础模型 (CPU)
    model_config = load_config(os.path.join(ckpt_path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model_raw = get_model(model_config, 'work_0605')
    model_raw.load_state_dict_from_path(os.path.join(ckpt_path, 'model.pt'))

    # 加载评估配置
    eval_config = load_config(eval_config_path)

    # aligner (共用)
    aligner_config = load_config(os.path.join(root, 'research/recognition/code/', 'run_v1', 'aligners/configs/none.yaml'))
    aligner = get_aligner(aligner_config)

    all_results = []

    if fabric.local_rank == 0:
        print("=" * 70)
        print("评估流水线加速方案综合对比")
        print(f"配置: {args.num_gpu} GPU, {args.precision}")
        print("=" * 70)

    # ===== 配置 1: 无 compile, BF16 (基线) =====
    if fabric.local_rank == 0:
        print(f"\n{'='*70}")
        print("配置 1/4: 无 compile, BF16 (基线)")
        print(f"{'='*70}")

    setup_start = time.time()
    import copy
    model_1 = copy.deepcopy(model_raw)
    model_1 = fabric.setup(model_1)
    setup_time_1 = time.time() - setup_start

    pipeline_1 = pipeline_from_name('infer_model_pipeline', model_1, aligner)
    pipeline_1.integrity_check(dataset_color_space='RGB')
    transform_1 = pipeline_1.make_test_transform()

    timing_1 = run_evaluation(fabric, eval_config, pipeline_1, transform_1, 'no_compile_bf16')
    timing_1['setup_time'] = setup_time_1
    timing_1['compile_time'] = 0
    timing_1['trt_build_time'] = 0
    timing_1['trt_load_time'] = 0
    all_results.append(timing_1)

    del model_1, pipeline_1
    torch.cuda.empty_cache()

    # ===== 配置 2: 预 compile (max-autotune) + BF16 =====
    if fabric.local_rank == 0:
        print(f"\n{'='*70}")
        print("配置 2/4: 预 compile (max-autotune) + BF16")
        print(f"{'='*70}")

    model_2 = copy.deepcopy(model_raw)

    # compile (在 fabric.setup 之前)
    compile_start = time.time()
    model_2 = torch.compile(model_2, mode='max-autotune')
    compile_call_time = time.time() - compile_start

    model_2 = fabric.setup(model_2)

    # warmup: 触发实际编译
    warmup_start = time.time()
    dummy = torch.randn(2, 3, 112, 112, device=fabric.device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model_2(dummy)
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    total_compile_time = compile_call_time + warmup_time

    if fabric.local_rank == 0:
        print(f"  compile 调用: {compile_call_time:.1f}s")
        print(f"  首次 forward (实际编译): {warmup_time:.1f}s")
        print(f"  compile 总耗时: {total_compile_time:.1f}s")

    pipeline_2 = pipeline_from_name('infer_model_pipeline', model_2, aligner)
    pipeline_2.integrity_check(dataset_color_space='RGB')
    transform_2 = pipeline_2.make_test_transform()

    timing_2 = run_evaluation(fabric, eval_config, pipeline_2, transform_2, 'compile_bf16')
    timing_2['setup_time'] = 0
    timing_2['compile_time'] = total_compile_time
    timing_2['trt_build_time'] = 0
    timing_2['trt_load_time'] = 0
    all_results.append(timing_2)

    del model_2, pipeline_2
    torch.cuda.empty_cache()

    # ===== 配置 3: 预 compile (max-autotune) + FP16 =====
    if fabric.local_rank == 0:
        print(f"\n{'='*70}")
        print("配置 3/4: 预 compile (max-autotune) + FP16")
        print(f"{'='*70}")

    model_3 = copy.deepcopy(model_raw).half()

    compile_start = time.time()
    model_3 = torch.compile(model_3, mode='max-autotune')
    compile_call_time_3 = time.time() - compile_start

    model_3 = model_3.cuda()
    # FP16 不用 fabric.setup (避免 precision 覆盖)，手动 DDP
    if args.num_gpu > 1:
        model_3 = torch.nn.parallel.DistributedDataParallel(model_3, device_ids=[fabric.local_rank])

    warmup_start = time.time()
    dummy_fp16 = torch.randn(2, 3, 112, 112, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        _ = model_3(dummy_fp16)
    torch.cuda.synchronize()
    warmup_time_3 = time.time() - warmup_start
    total_compile_time_3 = compile_call_time_3 + warmup_time_3

    if fabric.local_rank == 0:
        print(f"  compile 总耗时: {total_compile_time_3:.1f}s")

    # FP16 pipeline wrapper
    class FP16Pipeline:
        def __init__(self, model, model_config):
            self.model = model
            self.color_space = 'RGB'
            self._model_config = model_config

        def integrity_check(self, dataset_color_space, pipeline_color_space=None):
            pass

        def make_test_transform(self):
            from torchvision import transforms
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        def __call__(self, x):
            return self.model(x.half()).float()

    pipeline_3 = FP16Pipeline(model_3, model_config)
    transform_3 = pipeline_3.make_test_transform()

    timing_3 = run_evaluation(fabric, eval_config, pipeline_3, transform_3, 'compile_fp16')
    timing_3['setup_time'] = 0
    timing_3['compile_time'] = total_compile_time_3
    timing_3['trt_build_time'] = 0
    timing_3['trt_load_time'] = 0
    all_results.append(timing_3)

    del model_3, pipeline_3
    torch.cuda.empty_cache()

    # ===== 配置 4: TensorRT FP16 =====
    if not args.skip_trt:
        if fabric.local_rank == 0:
            print(f"\n{'='*70}")
            print("配置 4/4: TensorRT FP16")
            print(f"{'='*70}")

        # 只在 rank 0 构建 engine，其他 rank 等待
        engine_path = None
        trt_build_time = 0
        trt_load_time = 0

        if fabric.local_rank == 0:
            try:
                engine_path, trt_build_time, trt_load_time = build_or_load_trt_engine(
                    model_raw, batch_size=256, cache_dir='/tmp/trt_cache'
                )
                print(f"  TRT build: {trt_build_time:.1f}s, load: {trt_load_time:.1f}s")
                print(f"  Engine: {engine_path}")
            except Exception as e:
                print(f"  TRT 构建失败: {e}")
                import traceback
                traceback.print_exc()

        fabric.barrier()

        # 所有 rank 加载 engine
        engine_path = '/tmp/trt_cache/iresnet101_bs256_fp16.engine'
        if os.path.exists(engine_path):
            trt_pipeline = TRTPipeline(engine_path, batch_size=256, model_config=model_config)
            transform_4 = trt_pipeline.make_test_transform()

            timing_4 = run_evaluation(fabric, eval_config, trt_pipeline, transform_4, 'trt_fp16')
            timing_4['setup_time'] = 0
            timing_4['compile_time'] = 0
            timing_4['trt_build_time'] = trt_build_time
            timing_4['trt_load_time'] = trt_load_time
            all_results.append(timing_4)

            del trt_pipeline
        else:
            if fabric.local_rank == 0:
                print("  TRT engine 不存在，跳过")
    else:
        if fabric.local_rank == 0:
            print("\n[跳过 TRT 测试]")

    torch.cuda.empty_cache()

    # ===== 汇总 =====
    if fabric.local_rank == 0:
        print("\n\n")
        print("=" * 80)
        print("                    综合对比汇总")
        print("=" * 80)

        header = f"{'配置':<25} {'准备(s)':<10} {'特征提取(s)':<14} {'评估计算(s)':<14} {'总耗时(s)':<12} {'加速比'}"
        print(header)
        print("-" * 80)

        baseline_total = all_results[0]['total'] if all_results else 1
        for r in all_results:
            prep = r['compile_time'] + r['trt_build_time'] + r['trt_load_time']
            speedup = baseline_total / r['total'] if r['total'] > 0 else 0
            print(f"{r['config']:<25} {prep:<10.1f} {r['total_extract']:<14.1f} {r['total_compute']:<14.1f} {r['total']:<12.1f} {speedup:.2f}x")

        print("-" * 80)
        print("\n详细时间分解:")
        for r in all_results:
            print(f"\n  [{r['config']}]")
            print(f"    compile 生成/加载: {r['compile_time']:.1f}s")
            print(f"    TRT 构建: {r['trt_build_time']:.1f}s, 加载: {r['trt_load_time']:.1f}s")
            print(f"    各评估器:")
            for name, t in r['evaluators'].items():
                print(f"      {name}: extract={t['extract']:.1f}s, compute={t['compute']:.1f}s")

        # 保存 JSON
        output_path = os.path.join(os.path.dirname(__file__), 'benchmark_eval_pipeline_results.json')
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_path}")
        print("=" * 80)

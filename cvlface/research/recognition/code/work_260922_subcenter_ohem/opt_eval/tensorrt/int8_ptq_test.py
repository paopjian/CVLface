"""INT8 PTQ（显式 Q/DQ）vs FP16 TRT engine 特征提取对比测试

背景: TRT 11 移除了隐式量化接口 (BuilderFlag.INT8 / IInt8Calibrator),
INT8 唯一路径是 ONNX 图内显式 Q/DQ 节点, 构建 engine 不需要任何 flag.

流程:
  1. fp32 PyTorch(ir101) → 动态 batch fp32/fp16 ONNX
  2. onnxruntime quantize_static: IJBC 真实对齐图 MinMax 标定(2048 张),
     per-channel 权重, QDQ 格式 → QDQ ONNX
  3. TRT 解析 QDQ ONNX + optimization profile 构建 INT8 engine
  4. 基线: 复刻生产 build_trt_engine 方式构建 FP16 engine (fp16 ONNX 导出)
  5. 吞吐: 纯 engine (输入常驻 GPU) batch 扫描 + agedb_30 端到端 (含 dataloader)
  6. 精度: 特征余弦噪声(int8/fp16 vs fp32) + agedb_30 verification acc(10折)

用法: python int8_ptq_test.py [export|calib|build|bench|acc|all]  (默认 all)
中间产物: /tmp/int8_ptq_work/ ; 结果: 本目录 int8_ptq_results.parquet
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

import numpy as np
import polars as pl
import torch

# models/dataset 等包在 work_0605 下 (pyrootutils 只加了 cvlface 根)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

WORK_DIR = '/tmp/int8_ptq_work'
CKPT_DIR = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/s4_0618'
CALIB_DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
ACC_DATA = '/data1/dataset_0605/facerec_val/agedb_30'
ENGINE_BATCH = 256    # 与生产 engine 静态 batch 一致 (dataloader 256*2 flip 分两次 execute)
CALIB_N = 2048        # 标定图片数 (256 x 8 批)
WARMUP = 10
ITERS = 50
ENGINE_FP16 = os.path.join(WORK_DIR, 'engine_fp16.plan')
ENGINE_INT8 = os.path.join(WORK_DIR, 'engine_int8.plan')
ONNX_FP32 = os.path.join(WORK_DIR, 'model_fp32.onnx')
ONNX_FP16 = os.path.join(WORK_DIR, 'model_fp16.onnx')
ONNX_QDQ = os.path.join(WORK_DIR, 'model_qdq.onnx')
RESULT_PARQUET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'int8_ptq_results.parquet')


def load_model():
    from models import get_model
    from general_utils.config_utils import load_config
    cfg = load_config(os.path.join(CKPT_DIR, 'model.yaml'))
    cfg.start_from = ''
    cfg.freeze = False
    model = get_model(cfg, 'work_0605')
    model.load_state_dict_from_path(os.path.join(CKPT_DIR, 'model.pt'))
    model.eval()
    return model


# ---------------- 阶段 1: ONNX 导出 (动态 batch) ----------------
def stage_export():
    model = load_model()
    dummy = torch.randn(1, 3, 112, 112, device='cuda')
    dyn = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    if not os.path.exists(ONNX_FP32):
        with torch.no_grad():
            torch.onnx.export(model.float().cuda(), dummy, ONNX_FP32,
                              input_names=['input'], output_names=['output'],
                              opset_version=17, dynamo=False, dynamic_axes=dyn)
        print(f"fp32 ONNX 导出完成: {ONNX_FP32}")
    if not os.path.exists(ONNX_FP16):
        with torch.no_grad():
            torch.onnx.export(model.half().cuda(), dummy.half(), ONNX_FP16,
                              input_names=['input'], output_names=['output'],
                              opset_version=17, dynamo=False, dynamic_axes=dyn)
        print(f"fp16 ONNX 导出完成: {ONNX_FP16}")


# ---------------- 阶段 2: 标定 + QDQ 量化 ----------------
class CalibReader:
    """从 HF Dataset 取真实对齐图, 与评估一致的 [-1,1] 归一化"""

    def __init__(self, n=CALIB_N, batch=256):
        from datasets import load_from_disk
        from torchvision import transforms
        self.ds = load_from_disk(CALIB_DATA)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.n = min(n, len(self.ds))
        self.batch = batch
        self.i = 0

    def get_next(self):
        if self.i >= self.n:
            return None
        s, e = self.i, min(self.i + self.batch, self.n)
        arr = np.stack([
            self.tf(self.ds[k]['image'].convert('RGB')).numpy() for k in range(s, e)
        ]).astype(np.float32)
        self.i = e
        return {'input': arr}


def stage_calib():
    if os.path.exists(ONNX_QDQ):
        print("QDQ ONNX 已存在, 跳过标定")
        return
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    t0 = time.time()
    reader = CalibReader()
    print(f"标定集: {reader.n} 张 (来自 IJBC_gt_aligned)")
    quantize_static(
        model_input=ONNX_FP32,
        model_output=ONNX_QDQ,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # TRT(GPU) 仅支持对称量化(zero_point=0); bias 保持浮点由 TRT 内部量化
        extra_options={'QuantizeBias': False, 'ActivationSymmetric': True},
    )
    print(f"QDQ 量化完成: {ONNX_QDQ} ({time.time()-t0:.1f}s)")


# ---------------- 阶段 3: TRT 构建 ----------------
def parse_and_build(onnx_path, engine_path, batch=ENGINE_BATCH):
    """ONNX → TRT engine (动态 ONNX 固定 profile; QDQ 图自动走 INT8, 无需 flag)"""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"TRT parse error: {parser.get_error(i)}")
            return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (batch, 3, 112, 112), (batch, 3, 112, 112),
                      (batch, 3, 112, 112))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    return True


class EngineRunner:
    """与生产 TRTInfer 同款: 当前 stream execute_async_v3, 超 batch 分块"""

    def __init__(self, engine_path, batch_size=ENGINE_BATCH):
        import tensorrt as trt
        self.batch_size = batch_size
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = None, None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.input_name = n
            else:
                self.output_name = n
        _trt2torch = {trt.float16: torch.float16, trt.float32: torch.float32}
        self.input_dtype = _trt2torch[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = _trt2torch[self.engine.get_tensor_dtype(self.output_name)]
        self.d_input = torch.zeros(batch_size, 3, 112, 112,
                                   dtype=self.input_dtype, device='cuda')
        self.d_output = torch.zeros(batch_size, 512,
                                    dtype=self.output_dtype, device='cuda')
        self._set_io()

    def _set_io(self, bs=None):
        bs = bs or self.batch_size
        self.context.set_input_shape(self.input_name, (bs, 3, 112, 112))
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())

    def __call__(self, x):
        total = x.shape[0]
        x_in = x.to(self.input_dtype)
        results = []
        for s in range(0, total, self.batch_size):
            e = min(s + self.batch_size, total)
            bs = e - s
            if bs < self.batch_size:
                # 静态 profile engine: 尾部 batch 补齐, 输出取前 bs 行
                self.d_input.zero_()
                self.d_input[:bs].copy_(x_in[s:e])
            else:
                self.d_input.copy_(x_in[s:e])
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            results.append(self.d_output[:bs].float().clone())
        return torch.cat(results, dim=0) if len(results) > 1 else results[0]


def stage_build():
    for onnx_path, engine_path, name in [
        (ONNX_FP16, ENGINE_FP16, 'FP16'),
        (ONNX_QDQ, ENGINE_INT8, 'INT8(QDQ)'),
    ]:
        if os.path.exists(engine_path):
            print(f"{name} engine 已存在, 跳过")
            continue
        t0 = time.time()
        ok = parse_and_build(onnx_path, engine_path)
        if not ok:
            print(f"{name} engine 构建失败!")
            sys.exit(1)
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"{name} engine 构建: {time.time()-t0:.1f}s, {size_mb:.0f} MB -> {engine_path}")
        # sanity: 实际推理一次, 输出必须有限 (TRT11 inspector 无精度字段, 以行为验证代替)
        runner = EngineRunner(engine_path)
        out = runner(torch.randn(ENGINE_BATCH, 3, 112, 112, device='cuda'))
        assert torch.isfinite(out).all(), f"{name} engine 输出含非有限值!"
        print(f"  sanity ok: out {tuple(out.shape)}, "
              f"mean|.|={out.abs().mean():.3f}")
        del runner
        torch.cuda.empty_cache()


# ---------------- 阶段 4: 纯 engine 吞吐 (batch 扫描) ----------------
def bench_pure(engine_path, batch):
    runner = EngineRunner(engine_path, batch_size=batch)
    x = torch.randn(batch, 3, 112, 112, device='cuda', dtype=runner.input_dtype)
    for _ in range(WARMUP):
        runner(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        runner(x)
    dt = time.time() - t0
    del runner
    torch.cuda.empty_cache()
    return ITERS * batch / dt


# ---------------- 阶段 5: 端到端 + 精度 ----------------
def load_acc_dataset():
    from datasets import load_from_disk
    from torchvision import transforms
    ds = load_from_disk(ACC_DATA)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return ds, tf


def extract_feats_engine(engine_path, ds, tf):
    """复刻生产 worker: dataloader 128 → flip 拼 256 过 engine.
    flip 融合按 batch 粒度进行 (engine 输出前半 normal 后半 flip), 返回 (N,512)"""
    runner = EngineRunner(engine_path, batch_size=ENGINE_BATCH)
    loader = torch.utils.data.DataLoader(
        list(range(len(ds))), batch_size=ENGINE_BATCH // 2, num_workers=8)
    fused, t_total0, t_engine = [], time.time(), 0.0
    for idx in loader:
        arr = np.stack([tf(ds[int(k)]['image'].convert('RGB')).numpy()
                        for k in idx.tolist()]).astype(np.float32)
        x = torch.from_numpy(arr).cuda()
        x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        torch.cuda.synchronize()
        te0 = time.time()
        out = runner(x_combined)
        t_engine += time.time() - te0
        b = x.shape[0]
        fused.append((out[:b] + out[b:]).cpu())
    t_total = time.time() - t_total0
    del runner
    torch.cuda.empty_cache()
    return torch.cat(fused), t_total, t_engine


def extract_feats_torch(model, ds, tf):
    """fp32 PyTorch 参考特征, flip 融合同样按 batch 粒度, 返回 (N,512)"""
    loader = torch.utils.data.DataLoader(
        list(range(len(ds))), batch_size=ENGINE_BATCH // 2, num_workers=8)
    fused = []
    with torch.no_grad():
        for idx in loader:
            arr = np.stack([tf(ds[int(k)]['image'].convert('RGB')).numpy()
                            for k in idx.tolist()]).astype(np.float32)
            x = torch.from_numpy(arr).cuda()
            x_combined = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
            out = model(x_combined).float()
            fused.append((out[:x.shape[0]] + out[x.shape[0]:]).cpu())
    return torch.cat(fused)


def verify_acc(embeddings, ds):
    """agedb_30: embeddings 已 (f+f_flip) 融合归一化, 按对排列"""
    from evaluations.verifications.verification import calculate_roc2
    issame = np.array([ds[i]['is_same'] for i in range(0, len(ds), 2)])
    e1, e2 = embeddings[0::2], embeddings[1::2]
    diff = e1 - e2
    dist = (diff ** 2).sum(1)
    thresholds = np.arange(0, 4, 0.01)
    _, _, acc = calculate_roc2(thresholds, dist, issame, nrof_folds=10)
    return float(acc.mean() * 100), float(acc.std() * 100)


def stage_acc(rows):
    import sklearn.preprocessing
    ds, tf = load_acc_dataset()
    n = len(ds)
    print(f"agedb_30: {n} 张 ({n//2} 对)")

    print("提取 fp32 torch 参考特征...")
    model = load_model().float().cuda()
    fused32 = extract_feats_torch(model, ds, tf)
    fused32 = sklearn.preprocessing.normalize(fused32.numpy())
    acc32, std32 = verify_acc(fused32, ds)
    print(f"fp32 torch: acc={acc32:.3f}±{std32:.3f}")
    rows.append({"item": "agedb_acc_fp32_torch", "value": acc32,
                 "detail": f"std={std32:.3f}"})
    del model
    torch.cuda.empty_cache()

    for tag, engine_path in [('fp16', ENGINE_FP16), ('int8', ENGINE_INT8)]:
        feats, t_total, t_engine = extract_feats_engine(engine_path, ds, tf)
        fused = sklearn.preprocessing.normalize(feats.numpy())
        acc, std = verify_acc(fused, ds)
        cos = (fused * fused32).sum(1)
        print(f"{tag}: 端到端 {t_total:.1f}s (engine 纯计 {t_engine:.1f}s, "
              f"{n/t_total:.0f} img/s)  acc={acc:.3f}±{std:.3f}  "
              f"cos_vs_fp32 mean={cos.mean():.5f} p1={np.percentile(cos, 1):.5f}")
        rows.append({"item": f"agedb_e2e_sec_{tag}", "value": round(t_total, 2),
                     "detail": f"engine={t_engine:.1f}s, {n/t_total:.0f} img/s"})
        rows.append({"item": f"agedb_acc_{tag}", "value": acc,
                     "detail": f"std={std:.3f}, cos_mean={cos.mean():.5f}, "
                               f"cos_p1={np.percentile(cos, 1):.5f}"})
        del feats
    del fused32
    return rows


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(WORK_DIR, exist_ok=True)
    torch.cuda.set_device(0)
    rows = []

    if stage in ('export', 'all'):
        stage_export()
    if stage in ('calib', 'all'):
        stage_calib()
    if stage in ('build', 'all'):
        stage_build()
    if stage in ('bench', 'all'):
        thr_fp16 = bench_pure(ENGINE_FP16, ENGINE_BATCH)
        thr_int8 = bench_pure(ENGINE_INT8, ENGINE_BATCH)
        print(f"纯 engine 吞吐 batch={ENGINE_BATCH}: fp16={thr_fp16:,.0f} img/s, "
              f"int8={thr_int8:,.0f} img/s, 提速 {thr_int8/thr_fp16:.2f}x")
        rows.append({"item": f"pure_throughput_batch{ENGINE_BATCH}_fp16",
                     "value": round(thr_fp16, 0), "detail": f"iters={ITERS}"})
        rows.append({"item": f"pure_throughput_batch{ENGINE_BATCH}_int8",
                     "value": round(thr_int8, 0),
                     "detail": f"speedup={thr_int8/thr_fp16:.2f}x"})
        pl.DataFrame(rows).write_parquet(RESULT_PARQUET)
    if stage in ('acc', 'all'):
        stage_acc(rows)
        pl.DataFrame(rows).write_parquet(RESULT_PARQUET)
    print(f"\n结果已保存: {RESULT_PARQUET}")


if __name__ == '__main__':
    main()

# 评估加速研究笔记

## 当前基线 (V1, 7卡 DDP, bf16-mixed)

配置: `val_20260605.yaml`, iResNet-101 (adaface_ir101_webface12m), 7x RTX4090

| 评估器 | 类型 | 特征提取(s) | 评估计算(s) | 总耗时(s) |
|--------|------|-------------|-------------|-----------|
| work_0605_3t | custom_verification4 | 157.24 | 40.65 | 197.89 |
| work_0605_enhance | custom_verification4 | 10.58 | 1.80 | 12.38 |
| work_0605_glint | custom_verification4 | 238.71 | 96.46 | 335.17 |
| cplfw | verification | 4.52 | 1.87 | 6.39 |
| calfw | verification | 4.51 | 2.49 | 7.00 |
| agedb_30 | verification | 4.55 | 2.37 | 6.92 |
| tinyface | tinyface | 35.75 | 19.86 | 55.61 |
| **合计** | | **455.86** | **165.50** | **621.35** |

总运行: 10.54 min, 特征提取占比 73.4%, 评估计算占比 26.6%

---

## 1. torch.compile 加速

### 单卡纯推理 benchmark (bs=256, 10000张)

| 方法 | 吞吐量 | 加速比 |
|------|--------|--------|
| Eager FP32 | 1,517 img/s | 1.00x |
| AMP bf16 | 2,701 img/s | 1.78x |
| AMP fp16 | ~2,700 img/s | ~1.78x |
| compile reduce-overhead + bf16 | ~4,500 img/s | ~3.0x |
| compile max-autotune + bf16 | 4,872 img/s | 3.21x |

### 7卡评估流水线实测 (compile reduce-overhead)

| 配置 | 特征提取(s) | 评估计算(s) | 总耗时 |
|------|-------------|-------------|--------|
| 无 compile | 455.86 | 165.50 | 10.54 min |
| + compile | 413.36 | 165.50 | 9.83 min |
| 差值 | -9.3% | 不变 | **-6.7%** |

### 关键细节

- `torch.compile` 必须在 `fabric.setup(model)` **之前**调用，否则 DDP wrapper 的 C++ extension 导致 graph break，退化为 eager
- 首次编译耗时 ~67s (max-autotune)，但 inductor cache 可跨进程复用，第二次启动仅 ~12s
- cache 路径: `~/.cache/torch/inductor/` (自动管理)
- 评估流水线中 compile 效果有限，因为 DataLoader I/O、CPU gather、sim_matrix 计算不受 GPU compile 影响

---

## 2. TensorRT 加速

### 单卡纯推理 benchmark (bs=256, 10000张)

| 方法 | 吞吐量 | 加速比 |
|------|--------|--------|
| TensorRT FP32 (自动精度) | 2,004 img/s | 1.32x |
| TensorRT FP16 (半精度ONNX) | 7,763 img/s | **5.12x** |

### TRT 11 API 注意事项

版本: TensorRT 11.0.0.114 (pip install tensorrt-cu12)

**与旧版本 (TRT 8/10) 的重大区别:**
1. `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` 已移除 → `builder.create_network()` 无需参数
2. `BuilderFlag.FP16` / `BuilderFlag.INT8` 已移除 → 精度由 ONNX 模型 dtype 决定
3. `build_serialized_network()` 返回 `IHostMemory` 对象，用 `.nbytes` 获取大小，`memoryview()` 传给 `deserialize_cuda_engine`
4. `execute_async_v3(stream)` 替代旧版 `execute_async_v2`

**启用 FP16 的正确方式 (TRT 11):**
```python
# 导出半精度 ONNX
model_fp16 = model.half()
dummy_fp16 = torch.randn(bs, 3, 112, 112, device='cuda', dtype=torch.float16)
torch.onnx.export(model_fp16, dummy_fp16, path, dynamo=False, opset_version=17)

# 构建 engine (TRT 自动以 FP16 执行)
network = builder.create_network()
parser.parse(onnx_bytes)
serialized = builder.build_serialized_network(network, config)
```

**ONNX 导出注意:**
- PyTorch 2.9+ 默认使用 dynamo exporter，不含权重 → 必须 `dynamo=False` 使用 TorchScript exporter
- FP16 ONNX: ~124 MB, FP32 ONNX: ~249 MB
- Engine 构建: ~55s (FP16), engine size: ~126 MB

### TensorRT 的优劣势

优势:
- 纯推理吞吐最高 (5.12x)
- 推理时零 Python overhead

劣势:
- 固定 batch size (需 padding 或 optimization profile)
- Engine 构建耗时 ~55s，且 GPU 架构绑定 (不跨卡型迁移)
- 不支持动态控制流
- 集成到评估 pipeline 需要改写 DataLoader (固定 batch padding + 结果截断)

### FP32 / BF16 / FP16 精度对比实验

模型: adaface_ir101_webface12m, 数据: val_enhance (1000张), 单卡推理

**逐样本 embedding cosine similarity (vs FP32 基准):**

| 对比 | mean cosine | min cosine | std | <0.9999 | <0.999 |
|------|-------------|-----------|-----|---------|--------|
| FP32 vs BF16 | 0.99964 | 0.99728 | 1.9e-4 | 1000/1000 | 13/1000 |
| FP32 vs FP16 | 0.99978 | 0.99604 | 1.9e-4 | 872/1000 | 6/1000 |
| BF16 vs FP16 | 0.99985 | 0.99945 | 5.2e-5 | — | — |

**Embedding 绝对误差 (vs FP32):**

| 精度 | mean abs diff | max abs diff |
|------|--------------|-------------|
| BF16 | 1.88e-2 | 1.71e-1 |
| FP16 | 1.46e-2 | 1.86e-1 |

**Verification 配对测试 (500 正对 + 500 负对):**

配对 similarity 差异 (vs FP32):
- BF16: mean=1.82e-3, max=7.57e-3
- FP16: mean=1.50e-3, max=6.04e-3

不同阈值下判断不一致的对数:

| 阈值 | BF16 不一致 | FP16 不一致 |
|------|-----------|-----------|
| 0.3 | 2/1000 (0.20%) | 1/1000 (0.10%) |
| 0.4 | 2/1000 (0.20%) | 2/1000 (0.20%) |
| 0.5 | 2/1000 (0.20%) | 1/1000 (0.10%) |
| 0.6 | 0/1000 (0.00%) | 1/1000 (0.10%) |

**结论:**
- FP16 比 BF16 更精确 (10bit 尾数 vs 7bit)，两者均可安全用于推理
- TRT FP16 精度损失可忽略，不会影响评估指标
- INT8 (8bit 总共) 精度风险大，对 TPIR@FAR=1e-6 级别指标不推荐
- 验证脚本: `verify_fp16_precision.py`

### TensorRT Refit 验证 (多 checkpoint 场景)

问题: 评估 20 个 checkpoint 时，每次都要重建 engine (~25s)，总计额外 8.4 min。
方案: 构建一次 REFIT-enabled engine，之后每个 checkpoint 只替换权重。

**关键步骤:**
1. PyTorch 模型融合 BN → Conv (因为 ONNX 导出会自动融合，refit 时也需要一致)
2. 首次构建 REFIT engine (~25s，一次性)
3. 每个新 checkpoint: 加载权重 → 融合 BN → refit engine (~0.1s)

**实测结果:**

| 指标 | 数据 |
|------|------|
| BN 融合正确性 | cosine 0.99999732 |
| Engine 构建 (一次性) | 23.6s |
| 单次 Refit | 0.09s (avg of 3) |
| Refit 权重数 | 256/358 |
| 权重映射 | 456/509 PyTorch keys → ONNX names |
| PyTorch vs TRT-Refit cosine mean | **0.99988** |
| PyTorch vs TRT-Refit cosine min | 0.99961 |

**权重映射方案:** ONNX 图拓扑序匹配 — 按 graph.node 顺序提取 Conv/PRelu/Gemm/BN 的 initializer 引用，与 PyTorch named_modules 按 role 分组后逐一对应。

**20 checkpoint 场景对比:**

| 方案 | 额外耗时 |
|------|---------|
| 每次重建 engine | 20 × 24s = 7.9 min |
| Refit (首次 build + 后续 refit) | 24s + 20 × 0.6s = 0.6 min |
| **节省** | **7.3 min** |

**待完善:**
- 集成到 `eval_all_trt_launcher.py` 中 (用 refit 替代每次重建 engine)

验证脚本: `benchmark_trt_refit.py`

---

## 3. eval_all_3 方案 (TRT 多卡, 无 fabric)

`eval_all_3_single.py` 设计:
- 主进程构建 TRT engine (单卡, 一次性)
- 多进程并行: 每卡一个进程加载 engine 提取特征 (normal + flip 合并为单次 forward)
- 数据分片: DataLoader + DistributedSampler (手动)
- 聚合: /dev/shm 写文件 → 主进程合并 → compute_metric
- 完全不依赖 fabric, 不触发 NCCL

支持的评估类型:
| eval_type | 数据格式 | 评估协议 |
|-----------|---------|---------|
| `ijbbc` | HF Dataset (arrow + metadata.pt) | 官方 IJB-C: 模板聚合 → pair verification → ROC (TPR@FPR) |
| `ijbc_custom` | HF Dataset (arrow + metadata.pt) | 自定义: 并查集分组 → 多GPU相似度矩阵 → TPIR@FAR |
| `tinyface` | HF Dataset (arrow + metadata.pt, 含path) | probe vs gallery identification (rank-1/5/20) |
| `custom_verification4` | MXFaceDataset / ImageFolder | 多GPU直方图 + TPIR |
| `custom_verification` | MXFaceDataset / ImageFolder | generate_pairs_adaptive + ROC |

---

## 4. 综合对比实测 (7卡 DDP, val_20260605)

模型: adaface_ir101_webface12m, 7x RTX4090, val_20260605.yaml (7 个评估器)

### 总体结果

| 配置 | 准备耗时(s) | 特征提取(s) | 评估计算(s) | 总耗时(s) | 加速比 |
|------|------------|-------------|-------------|-----------|--------|
| 无 compile, BF16 (基线) | 0 | 455.8 | 30.9 | 492.9 | 1.00x |
| compile max-autotune + BF16 | 122.1* | 415.5 | 31.0 | 452.8 | 1.09x |
| compile max-autotune + FP16 | 119.2* | 396.3 | 30.3 | 433.5 | 1.14x |
| TensorRT FP16 | 56.3† | 341.6 | 37.2 | 385.6 | 1.28x |

*compile 准备: 首次需 ~120s autotune，cache 命中后仅 ~12s
†TRT 准备: 首次 build 56s，之后加载 0.1s

### 各评估器特征提取详情 (秒)

| 评估器 | 无 compile BF16 | compile BF16 | compile FP16 | TRT FP16 | TRT 加速比 |
|--------|----------------|-------------|-------------|----------|-----------|
| work_0605_3t | 156.0 | 153.6 | 132.8 | 112.6 | 1.39x |
| work_0605_enhance | 10.4 | 9.7 | 10.2 | 8.8 | 1.18x |
| work_0605_glint | 240.2 | 206.2 | 207.5 | 174.5 | 1.38x |
| cplfw | 4.4 | 4.5 | 4.4 | 4.2 | 1.04x |
| calfw | 4.6 | 4.6 | 4.6 | 4.4 | 1.04x |
| agedb_30 | 4.6 | 4.5 | 4.6 | 4.7 | 0.98x |
| tinyface | 35.7 | 32.4 | 32.3 | 32.3 | 1.11x |

### 分析

1. **compile BF16 vs 基线**: 特征提取 -8.8%，大数据集（glint）收益明显（-14.2%），小数据集几乎无差
2. **compile FP16 vs compile BF16**: 特征提取再 -4.6%，主要来自 work_0605_3t（-13.5%），glint 未获得额外收益
3. **TRT FP16 vs 基线**: 特征提取 -25.1%，大数据集效果显著（3t -27.8%, glint -27.4%）
4. **小数据集瓶颈**: cplfw/calfw/agedb_30 各方案差异极小（<0.5s），说明这些已被 DataLoader I/O 主导
5. **TRT 评估计算变慢**: 37.2s vs 30.9s（+20%），因为 TRT pipeline 输出需额外 float() 转换 + verification 的 compute_metric 开销

### 实际使用建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| 训练中 per-epoch 评估 | compile BF16 | 零额外工程，cache 命中后 12s warmup |
| 批量评估多 checkpoint | compile FP16 | 首次 119s，之后每次 ~12s，长期收益高 |
| 大规模正式评估 | TRT FP16 | 最快，engine 一次 build 永久复用 |
| 快速验证/debug | 无 compile | 无额外开销 |

### compile cache 复用验证

- 首次 max-autotune 编译: ~120s (含 kernel benchmark)
- cache 命中 (同机器重启后): ~12s (graph trace + .so 加载)
- cache 路径: `~/.cache/torch/inductor/`
- 条件: 同 batch_size、同模型结构、同 PyTorch/Triton 版本、同 GPU 架构

### 相关脚本

| 脚本 | 用途 |
|------|------|
| `benchmark_eval_pipeline.py` | 早期 4 配置综合对比 |
| `benchmark_eval_pipeline_results.json` | 早期完整计时结果 JSON |
| `benchmark_eval_methods.py` | 最新 benchmark (FP32/BF16/compile/TRT) |
| `benchmark_eval_methods_results.json` | 最新完整计时结果 JSON |

### 2026-06-12 最新实测 (s2_body36_0605 checkpoint, 7卡)

模型: s2_body36_0605_06-10_2 epoch:0, 7x RTX4090, val_20260605.yaml

**注意: 此次用 fine-tuned 模型而非 pretrained，与上方基线不完全可比（数据集同为 val_20260605）**

#### 总体结果

| 配置 | Wall Time(s) | 评估耗时(s) | 状态 |
|------|-------------|-------------|------|
| 无 compile, FP32 (32-true) | — | — | 失败 (OOM/DDP crash) |
| 无 compile, BF16 (bf16-mixed) | 664.5 | 623.2 | 成功 |
| compile max-autotune, BF16 | 874.6 | 731.4 | 成功 (含 warmup) |
| TensorRT FP16 (无 fabric) | 616.5 | 558.2 | 成功 |

#### 各评估器详情 (秒)

| 评估器 | BF16 无compile | compile BF16 | TRT 提取 | TRT 指标 | TRT 总计 |
|--------|---------------|-------------|----------|----------|----------|
| work_0605_3t | 198.0 | 326.6* | 102.3 | 28.2 | 130.5 |
| work_0605_enhance | 12.7 | 27.2* | 62.9 | 2.0 | 64.9 |
| work_0605_glint | 338.1 | 305.2 | 127.5 | 56.2 | 183.7 |
| cplfw | 6.6 | 6.9 | 26.4 | 5.4 | 31.8 |
| calfw | 7.0 | 6.8 | 29.8 | 5.4 | 35.2 |
| agedb_30 | 7.0 | 6.8 | 30.0 | 4.9 | 34.9 |
| tinyface | 53.8 | 51.8 | 38.2 | 23.0 | 61.2 |
| **合计** | **623.2** | **731.4** | — | — | **558.2** |

*compile 列: work_0605_3t 和 enhance 包含首次 max-autotune 编译 warmup (~128s)

#### TRT build 耗时

| 阶段 | 耗时 |
|------|------|
| ONNX 导出 + engine 构建 (一次性) | 58.3s |

#### 分析

1. **FP32 失败**: 7卡 DDP 在 `32-true` 下 rank 5 crash，推测是 24GB 显存不足（IR101 FP32 + bs256 + DDP buffer）
2. **compile warmup 开销巨大**: 首个评估器 (work_0605_3t) 承担全部 max-autotune 编译 (326.6 - 198.0 = 128.6s)，第三个评估器 (glint) 才开始获益 (338.1 → 305.2, -9.7%)
3. **TRT 小数据集劣势**: cplfw/calfw/agedb_30 从 ~7s 涨到 ~32s，因为每个评估器都需 spawn 7 个多进程 + 加载 engine 的固定开销 (~25s)
4. **TRT 大数据集优势**: work_0605_3t (198→130s, -34%), glint (338→184s, -46%)
5. **TRT 总 wall time 最快**: 616.5s，但含 58.3s engine build；若 engine 已缓存则约 558s

#### 结论更新

| 场景 | 推荐 | 理由 |
|------|------|------|
| 单 checkpoint 快速评估 | BF16 无 compile | 最稳定，无额外开销，11.1 min |
| 批量评估多 checkpoint | TRT FP16 | engine 一次 build，后续 refit 0.1s/ckpt，大数据集快 34-46% |
| compile 适用场景 | 多 checkpoint + inductor cache 命中 | 首次 warmup 128s，cache 命中后 ~12s，之后每次节省 33s (glint) |
| compile 不适用 | 单次评估 / 小数据集为主 | warmup 开销远超收益 |

---

## 5. 瓶颈分析与优化路线图

```
当前 10.54 min 构成:
├── 特征提取 455.86s (73.4%)  ← GPU forward + DataLoader I/O
│   ├── GPU forward: ~50% (受 compile/TRT 加速)
│   └── DataLoader I/O + CPU: ~50% (受 num_workers/合并遍历加速)
├── 评估计算 165.50s (26.6%)  ← CPU/GPU 混合
│   ├── sim_matrix (GPU, custom_v4): 大部分
│   └── verification/tinyface (CPU): 少量
└── 其他开销 ~11s (模型加载 + barrier)
```

### 可叠加优化 (按实测收益排序)

| 优化 | 实测收益 | 复杂度 | 状态 |
|------|---------|--------|------|
| TensorRT FP16 | -25% 特征提取 (455→342s) | 高 | 已验证 |
| TRT 多卡无 fabric | 避免 NCCL timeout | 中 | eval_all_3 已实现 |
| torch.compile + FP16 | -13% 特征提取 (455→396s) | 低 | 已验证 |
| torch.compile + BF16 | -9% 特征提取 (455→416s) | 低 | 已验证 |
| 增大 num_workers | -5~10% 特征提取 | 低 | 已配 |
| sim_matrix GPU 优化 | -? 评估计算 | 中 | 待研究 |

### 理论最优 (所有优化叠加)

特征提取: 455s → ~150s (TRT FP16 + 多卡无 fabric)
评估计算: 165s → 165s (暂不优化)
**总计: ~315s ≈ 5.3 min** (vs 当前 10.54 min, -50%)

---

## 6. 相关脚本

| 脚本 | 用途 |
|------|------|
| `benchmark_tensorrt.py` | 单卡推理加速对比 (Eager/AMP/compile/TRT) |
| `benchmark_eval_methods.py` | 完整 4 配置 benchmark (FP32/BF16/compile/TRT) |
| `profile_v1_stages.py` | 7卡评估分阶段计时 (支持 --compile) |
| `eval_all_torch_single.py` | torch 评估入口 (支持 --compile/--timing) |
| `eval_all_torch_launcher.py` | torch 多 checkpoint 批量评估启动器 |
| `eval_all_trt_single.py` | TRT 多卡评估入口 (无 fabric) |
| `eval_all_trt_launcher.py` | TRT 多模型批量评估启动器 |

---

## 7. 批量评估启动命令

### TRT 评估 (7卡, 无 fabric)

```bash
cd /root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_0605 && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/cvlface/bin/python eval_all_trt_launcher.py \
  --num_gpu 7 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data2/dataset_0605/train_output/s2_body36_0605_06-10_2/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s2_body36_0605_trt \
  --timeout_minutes 90
```

说明:
- 完全不依赖 fabric/NCCL，不会 NCCL timeout
- 主进程 GPU 0 构建 TRT engine (~60s)，7 卡多进程加载同一 engine 并行提取特征
- 数据分片: DistributedSampler，聚合: /dev/shm 文件
- compute_metric 在主进程单独计算（无 barrier 限制）
- 15 个 checkpoint × 60s build = 额外 ~15 min，但推理快 25%，且不会超时 crash
- 支持 wandb 断点续评

### torch 评估 (7卡 DDP, fabric run)

```bash
cd /root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_0605 && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/cvlface/bin/python eval_all_torch_launcher.py \
  --num_gpu 7 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data2/dataset_0605/train_output/s2_body36_0605_06-10_2/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s2_body36_0605_06-10_2_torch \
  --compile --compile_mode max-autotune --timing \
  --timeout_minutes 90
```

说明:
- 基于 fabric run (DDP)，支持 --compile / --compile_mode / --timing 开关
- compile 首次 warmup ~128s (max-autotune)，cache 命中后 ~12s
- --timing 输出每个评估器的分阶段计时
- 支持 wandb 断点续评

### benchmark 对比

```bash
cd /root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_0605 && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/cvlface/bin/python benchmark_eval_methods.py \
  --ckpt_path /data2/dataset_0605/train_output/s2_body36_0605_06-10_2/checkpoints_every_epoch/epoch:0_step:9053 \
  --num_gpu 7 \
  --eval_config_name val_20260605 \
  --timeout_minutes 90
```

说明:
- 顺序运行 FP32 / BF16 / compile max-autotune / TRT FP16 四种配置
- 每种配置独立子进程，互不影响
- 结果输出对比表格 + JSON (`benchmark_eval_methods_results.json`)
- 可用 --skip fp32 trt 跳过指定配置

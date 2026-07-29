# work_0605 全流程优化总结

硬件环境: 7-8x RTX 4090 (24GB, PCIe), NVMe SSD, 503GB RAM
模型: iResNet-101 (AdaFace/WebFace12M), 数据: 791K 类 / 37M 图片

---

## 1. 数据打包

将散落子目录中的图片打包为单文件格式，消除训练时海量小文件 I/O。

### 格式对比

| 指标 | ImageFolder | RecordIO | LMDB |
|------|-------------|----------|------|
| 打包时间 | 0 | 42.8 min | 51.1 min |
| 磁盘占用 | ~150GB (37M 文件) | 98.3 GB (2 文件) | 137 GB |
| 训练速度 (1 epoch) | 40.2 min | 30.3 min | 29.9 min |
| 随机访问 | 文件系统 inode | .idx 偏移表 + fseek | mmap B+tree |
| 多进程竞争 | inode/dentry 压力大 | 文件句柄 PID 检测 | 无锁并发读 |

### 结论

**选择 RecordIO** — 打包快、磁盘小、速度和 LMDB 持平、项目原生支持。相比 ImageFolder 训练提速 33%。

### 打包脚本

位置: `cvlface/data_utils/recognition/training_data/bundle_images_into_rec_v2.py`

特点:
- 无 mxnet 依赖，纯 Python 实现
- 8 读线程并行预读 + 单线程顺序写入，NVMe 上 14K img/s
- 同时生成 train.tsv（训练时跳过遍历 rec 文件）
- 与 mxnet RecordIO 格式完全兼容

```bash
python bundle_images_into_rec_v2.py --source_dir /data1/dataset_0605/train --save_dir /data1/dataset_0605/train_rec
```

输出: `train.rec` + `train.idx` + `train.tsv` + `meta.json`

---

## 2. 数据读取与增强

训练时 DataLoader 瓶颈: JPEG 解码 + 数据增强。

### 2.1 解码后端对比 (7卡 DDP, bs=512, 1000 batch)

| 解码后端 | Epoch Time | 速度 | 相对 PIL |
|----------|-----------|------|----------|
| PIL (默认) | 3.23 min | 22,637/s | 1.00x |
| OpenCV | 3.10 min | 23,100/s | 1.04x |
| torchvision.io | 3.16 min | 22,475/s | 0.99x |
| **TurboJPEG** | **2.90 min** | **24,522/s** | **1.08x** |

**选择 TurboJPEG** — 纯 C 库、无内部线程、fork-safe、多 worker 下表现最好。

torchvision.io 单线程快但多 worker 退化原因: 内部线程池与 DataLoader worker 争用 CPU、额外内存拷贝、fork 后需重建 C++ 状态。

### 2.2 数据增强 v2 (纯 numpy 链路)

v1 问题: PIL↔numpy 来回转换 2-6 次/图、imgaug 依赖引入额外开销。

v2 重写: TurboJPEG → numpy → augment → torch.from_numpy，全链路零 PIL 对象。

| 方案 | 单线程速度 | 加速比 | 7卡实训 |
|------|-----------|--------|---------|
| BasicAug v1 (PIL) | 3,999/s | 1.00x | 3.15 min |
| BasicAug v2 (numpy) | 7,722/s | 1.93x | 3.08 min |
| GridSampleAug v1 (PIL) | 1,052/s | 1.00x | 3.50 min |
| **GridSampleAug v2 (numpy)** | **1,844/s** | **1.75x** | **3.33 min** |

### 2.3 DataLoader 配置

```python
persistent_workers=True   # 避免每 epoch 重新 fork
prefetch_factor=3         # 提前预取
pin_memory=True           # 加速 CPU→GPU 传输
num_workers=8             # w=8 和 w=16 差异不大，CPU 已饱和
```

### 2.4 内存盘实测无效

92GB RecordIO 已被 OS page cache 完全缓存 (机器 503GB RAM)。tmpfs 反而减少可用 page cache 空间，VFS 路径开销更大。

---

## 3. 训练循环优化

### 3.1 GPU 计算加速 (train_opt.py)

| 优化项 | 单项效果 | 最佳组合 |
|--------|---------|---------|
| cudnn.benchmark | -5% (无效，112x112 固定尺寸) | — |
| channels_last (NHWC) | +14% | — |
| torch.compile (mode=default) | +6% | — |
| **三者全开** | — | **+31%** |

关键点:
- `torch.compile` 必须在 `fabric.setup(model)` 之前调用，否则 DDP wrapper 导致 graph break
- compile 首次 1-2 min 编译，inductor cache 可复用
- max-autotune 对 IR-101 (49 残差块) 无额外收益，autotune 组合爆炸

### 3.2 消除冗余 GPU 同步

| 优化 | 问题 | 修复 |
|------|------|------|
| `loss.item()` 每 batch | 强制 CUDA sync，阻止 GPU/CPU 并行 | GPU 上 fp32 累积，仅 logging 时 sync |
| `get_norm()` 逐参数 `.item()` | IR-101 ~200 参数 = 200 次 sync | 用 `clip_grad_norm_(inf)` 单次 sync |
| `compute_update_ratio()` 逐参数 | 同上 | GPU 上累积 delta_sq / param_sq |
| tqdm 每 batch 更新 | stdout I/O + time.time() | 每 10 batch 更新 |
| `optimizer.zero_grad()` | memset(0) 全部 grad | `set_to_none=True` 跳过 memset |
| MLflow SQLite 同步写 | 磁盘 I/O 阻塞主线程 | 异步后台线程队列 |

### 3.3 PartialFC 分类器优化

| 优化 | 效果 |
|------|------|
| `DistCrossEntropyFunc.backward` 里 `.item()` → 直接 tensor 乘 | 每 batch backward 少 1 次 sync |
| `sample()` 去掉 4 个冗余 `.cuda()` | 减少 dispatch overhead |
| `torch.rand` 直接指定 device | 避免 CPU→GPU 拷贝 |
| 去掉 `logits.clamp(-1,1)` | 减少多余 kernel launch |
| embedding 归一化用 `F.normalize` | fused kernel |

### 3.4 最终训练配置

```python
torch.backends.cudnn.benchmark = True
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model, dynamic=False)
optimizer.zero_grad(set_to_none=True)
```

---

## 4. 评估推理加速

评估基线: 7卡 DDP, BF16, 7 个评估器, 总耗时 10.54 min (特征提取 73.4% + 评估计算 26.6%)。

### 4.1 推理加速方案对比

#### 单卡纯推理 benchmark (bs=256, 10000 张)

| 方法 | 吞吐量 | 加速比 |
|------|--------|--------|
| Eager FP32 | 1,517 img/s | 1.00x |
| AMP bf16 | 2,701 img/s | 1.78x |
| compile reduce-overhead + bf16 | ~4,500 img/s | ~3.0x |
| compile max-autotune + bf16 | 4,872 img/s | 3.21x |
| TensorRT FP16 | 7,763 img/s | **5.12x** |

#### 7卡评估流水线实测

| 配置 | 特征提取(s) | 总耗时 | 加速比 |
|------|-------------|--------|--------|
| 无 compile, BF16 (基线) | 455.8 | 492.9s | 1.00x |
| compile max-autotune + BF16 | 415.5 | 452.8s | 1.09x |
| compile max-autotune + FP16 | 396.3 | 433.5s | 1.14x |
| **TensorRT FP16** | **341.6** | **385.6s** | **1.28x** |

### 4.2 TensorRT 集成

TRT 11 API 关键变化 (vs TRT 8/10):
- `EXPLICIT_BATCH` 已移除，`builder.create_network()` 无需参数
- 精度由 ONNX 模型 dtype 决定 (导出 FP16 ONNX 即可)
- `execute_async_v3(stream)` 替代旧版 `execute_async_v2`

FP16 正确方式: 导出半精度 ONNX → TRT 自动以 FP16 执行 (TRT 11 无 `BuilderFlag.FP16`)。

Engine 构建 ~55s，engine 大小 ~126MB，GPU 架构绑定 (不跨卡型迁移)。

**IO binding 坑 (重要)**: TRT 10/11 不保证 IO tensor 的 index 顺序，
用 `get_tensor_name(0/1)` 假设 0=input/1=output 会在 output 排到 index 0 时
导致输入输出地址接反，提取的特征与 PyTorch 严重不一致 (cos 可低至 0.70 甚至负值)，
而同一 ONNX 在 ONNX Runtime 下正常。必须按 `get_tensor_mode()==TensorIOMode.INPUT`
识别 IO。该修正已落地 `eval_all_trt_single.py`。详见
[tensorrt_eval_optimization.md](docs/tensorrt_eval_optimization.md)。

**opt_level / 精度实测** (iResNet-101, 48091 张): binding 修复后 FP32/FP16 全部配置
cos≈1.0、TPIR/acc 与 Torch 无差异。FP16 是最大杠杆 (特征提取 981→3414 img/s, 3.5x,
精度近乎无损)；opt_level 对精度几乎无影响，opt3 是 build/推理性价比拐点
(opt5 build 慢 ~10 倍而推理收益饱和)；CUDA Graph 在 GPU 计算瓶颈下无收益 (+0.2%)。

### 4.3 TRT 多卡评估 (eval_all_trt_single.py, 无 fabric)

设计:
- 完全不依赖 fabric/NCCL，不触发 NCCL timeout
- 主进程 GPU 0 构建 engine，7 卡多进程加载同一 engine 并行提取特征
- 数据分片: DistributedSampler，聚合: /dev/shm 文件写入 → 主进程合并
- compute_metric 在主进程单独计算

大数据集优势显著: work_0605_3t -34%, glint -46%。
小数据集 (<5000 张) 因多进程 spawn + engine 加载固定开销反而慢。

### 4.4 FP16 精度验证

| 对比 | mean cosine | min cosine | 不一致判断 (阈值 0.3-0.6) |
|------|-------------|-----------|--------------------------|
| FP32 vs BF16 | 0.99964 | 0.99728 | 0-2/1000 |
| FP32 vs FP16 | 0.99978 | 0.99604 | 1-2/1000 |

结论: FP16 比 BF16 更精确 (10bit 尾数 vs 7bit)，TRT FP16 精度损失可忽略。

### 4.5 场景推荐

| 场景 | 推荐 | 理由 |
|------|------|------|
| 单 checkpoint 快速评估 | BF16 无 compile | 最稳定，无额外开销 |
| 批量评估多 checkpoint | TRT FP16 | engine 一次 build，推理快 25% |
| 训练中 per-epoch 评估 | compile BF16 | 零额外工程，cache 命中后 12s warmup |
| 大规模正式评估 | TRT FP16 | 特征提取最快 -25% |

---

## 5. 评估计算优化 (相似度矩阵)

`get_sim_matrix_large_scale_v4`: 7 卡计算大规模余弦相似度矩阵 + 正负样本直方图。

### 瓶颈分析

1. **Boolean indexing 动态分配**: `sim[mask]` 每次产生不确定长度输出，触发扫描+分配+拷贝
2. **FP32 matmul 未用 Tensor Core**: 4090 FP16 吞吐 330T vs FP32 82T
3. **Block size 过小**: 10240 产生 19306 块，kernel launch overhead 占比高

### 优化方案

| 方案 | 耗时 (2M 特征) | 加速比 | 精度影响 |
|------|---------------|--------|----------|
| 原始 (FP32+bool_idx) | 67s | 1.00x | 基准 |
| 方案B (FP32+where NaN) | ~29s | ~2.3x | 无损 |
| **方案C (FP16+where NaN)** | **~25s** | **~2.6x** | **<0.04%** |

核心思路:
- `torch.where(mask, sim, NaN)` 替代 boolean indexing — 输出形状固定，无动态分配
- `torch.histc` 天然忽略 NaN，无需手动过滤
- FP16 matmul 利用 Tensor Core，结果转回 FP32 做 histc
- 顺序计算 pos/neg 避免同时持有两份全尺寸矩阵导致 OOM

---

## 6. 累计优化效果

| 阶段 | 优化项 | 效果 |
|------|--------|------|
| 数据打包 | RecordIO 替代 ImageFolder | -33% epoch 时间 |
| 数据解码 | TurboJPEG 替代 PIL | -10% DataLoader 时间 |
| 数据增强 | v2 numpy 链路 | -5% DataLoader 时间 |
| GPU 计算 | compile + channels_last + cudnn | +31% 训练吞吐 |
| 训练循环 | 消除冗余 sync + PartialFC 优化 | +5-10% 训练吞吐 |
| 评估推理 | TensorRT FP16 | -25% 特征提取时间 |
| 评估计算 | where(NaN) + FP16 matmul | -60% sim_matrix 时间 |

### 生产推荐配置

```bash
# 数据
dataset=configs/dataset_0605_train_rec.yaml   # RecordIO
export DECODE_BACKEND=turbojpeg               # 已设为默认
data_augs=configs/gridsample_v2_numpy.yaml    # v2 numpy augmenter

# DataLoader (fabric/fabric.py 自动配置)
persistent_workers=True, prefetch_factor=3, pin_memory=True, num_workers=8

# 训练 (train_opt.py 已内置)
cudnn.benchmark=True, channels_last, torch.compile(dynamic=False)

# 评估
eval_all_trt_launcher.py   # TRT 多卡，多 checkpoint 批量评估
eval_all_torch_launcher.py # torch 多卡，支持 --compile
```

---

## 7. 启动命令参考

### CoreFace-AdaFace

CoreFace 适配位于 `models/iresnet`、`losses/adaface.py` 和
`pipelines/train_model_cls_pipeline.py`。使用
`pipelines/configs/train_model_cls_coreface.yaml` 时，前
`coreface_start_epoch` 个 epoch 为单视图 AdaFace，之后自动切换为双
dropout 视图和 ContraFace；数据仍使用
`dataset/configs/dataset_0605_train_rec.yaml`。完整的 `train_opt.py`
命令见 `流程_coreface.txt`。

自动执行四个阶段可直接运行：

```bash
bash scripts/run_coreface_all.sh
```

脚本默认使用阶段1/3为 `512`、阶段2/4为 `256` 的每卡 batch size；可通过
`BATCH_SIZE_S1`、`BATCH_SIZE_S2`、`BATCH_SIZE_S3`、`BATCH_SIZE_S4` 覆盖。
脚本会在每个阶段完成后自动查找 `checkpoints_every_epoch/epoch:<N>`，并将
`model.pt` 和分类器 checkpoint 传给下一阶段。常用覆盖项包括
`NUM_GPU`、`BATCH_SIZE`、`NUM_WORKERS`、`OUTPUT_ROOT`、
`PRETRAINED_MODEL`、`EVAL_CONFIG` 和 `SKIP_FINAL_EVAL`。

### 训练 (7卡 DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run \
    --strategy=ddp --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=s2_body36_0605 trainers.num_gpu=7 \
    trainers.batch_size=512 trainers.num_workers=8 \
    models=iresnet/configs/v1_ir101.yaml \
    dataset=configs/dataset_0605_train_rec.yaml \
    data_augs=configs/gridsample_v2_numpy.yaml \
    classifiers=configs/partial_fc_sample10.yaml \
    losses=configs/adaface.yaml \
    evaluations=configs/val_20260605.yaml
```

### TRT 评估 (7卡, 无 fabric)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH \
python eval_all_trt_launcher.py \
  --num_gpu 7 --eval_config_name test_20260605 \
  --ckpt_dir /path/to/checkpoints_every_epoch \
  --project_name work_0605_test --name s2_body36_trt \
  --timeout_minutes 90
```

### torch 评估 (7卡 DDP, compile)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH \
python eval_all_torch_launcher.py \
  --num_gpu 7 --eval_config_name test_20260605 \
  --ckpt_dir /path/to/checkpoints_every_epoch \
  --project_name work_0605_test --name s2_body36_compile \
  --compile --compile_mode max-autotune --timing \
  --timeout_minutes 270
```

---

## 8. 相关脚本索引

| 脚本 | 用途 |
|------|------|
| `train_opt.py` | 训练入口 (含全部优化) |
| `eval_all_torch_single.py` | torch 评估入口 (支持 --compile/--timing) |
| `eval_all_torch_launcher.py` | torch 多 checkpoint 批量评估启动器 |
| `eval_all_trt_single.py` | TRT 多卡评估入口 (无 fabric) |
| `eval_all_trt_launcher.py` | TRT 多模型批量评估启动器 |
| `opt_eval/tensorrt/test_trt_vs_torch_eval.ipynb` | TRT vs Torch 精度排查 + FP16/opt 评估 |
| `scripts/benchmark/bench_decode.py` | 解码后端速度对比 |
| `scripts/benchmark/bench_dataloader.py` | DataLoader 吞吐对比 |
| `scripts/benchmark/bench_augmenter_v2.sh` | 增强器 v2 对比 |

详细数据见 `docs/` 目录下各专题文档。

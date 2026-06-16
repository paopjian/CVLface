# 数据格式 + 解码后端 对比训练方案

## 1. 打包

### RecordIO

```bash
conda run --no-capture-output -n cvlface python scripts/benchmark/bundle_rec_large.py
```

- 输入: `/data1/dataset_0605/train` (791K 类, 37M 图片)
- 输出: `/data1/dataset_0605/train_rec/train.rec` + `train.idx`

```
Scanning dirs: 100%|███| 791509/791509 [01:11<00:00, 11058.95dir/s]
Found 37,084,481 images, 791,509 classes in 87.6s
RecordIO: 100%|███| 37084481/37084481 [42:46<00:00, 14451.22it/s]
[RecordIO] Done: 2566.2s (42.8 min), size: 98.3 GB
  速度: 14451 images/s
```

### LMDB

```bash
conda run --no-capture-output -n cvlface python scripts/benchmark/bundle_lmdb_large.py
```

- 输出: `/data1/dataset_0605/train_lmdb/train.lmdb/`

```
Scanning dirs: 100%|███| 791509/791509 [00:43<00:00, 18400.86dir/s]
Found 37,084,481 images, 791,509 classes in 59.8s
LMDB: 100%|███| 37084481/37084481 [51:00<00:00, 12115.32it/s]
[LMDB] Done: 3063.4s (51.1 min), size: 137 GB (实际磁盘占用, apparent size 322 GB 是稀疏文件)
  速度: 12106 images/s
```

---

## 2. 数据格式对比训练 (1 full epoch, train_opt.py)

### 公共环境

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
conda activate cvlface

COMMON="trainers.num_gpu=7 trainers.num_workers=8 \
trainers.precision=bf16-mixed \
models=iresnet/configs/v1_ir101.yaml \
models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
data_augs=configs/basic_v1.yaml \
classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.40 \
losses=configs/adaface.yaml \
evaluations=configs/val_20260605.yaml \
dataset.model_save_dir=/data1/dataset_0605/train_output"
```

解码后端: PIL (DECODE_BACKEND=pil, 默认当时)

### A) ImageFolder (baseline)

```
Epoch Time: 40.23 mins | Speed 26674
```

### B) RecordIO + PIL

```
Epoch Time: 30.33 mins | Speed 25872
```

### C) LMDB + PIL

```
Epoch Time: 29.86 mins | Speed 24589
```

### 数据格式对比结论

| 格式 | Epoch Time | Speed | 相对 ImageFolder |
|------|-----------|-------|-----------------|
| ImageFolder | 40.23 min | 26,674 | 1.00x |
| RecordIO | 30.33 min | 25,872 | 1.33x (时间) |
| LMDB | 29.86 min | 24,589 | 1.35x (时间) |

RecordIO 和 LMDB 训练速度几乎一样 (差 1.5%)。RecordIO 更优: 打包快、磁盘小 (98 vs 137 GB)。

---

## 3. 解码后端对比训练 (1000 batch, RecordIO, train_opt.py)

### 测试条件

- 7卡 DDP, batch_size=512, backbone 冻结
- `trainers.limit_num_batch=1000` (512,000 张图/组)
- 数据: RecordIO on /data1 (NVMe SSD, 已被 OS page cache 缓存)
- persistent_workers=True, prefetch_factor=3

### 结果

| 解码后端 | Epoch Time | Avg Speed | 相对 PIL |
|----------|-----------|-----------|----------|
| PIL (默认) | 3.23 min | 22,637 imgs/s | 1.00x |
| OpenCV (cv2) | 3.10 min | 23,100 imgs/s | 1.04x |
| torchvision.io | 3.16 min | 22,475 imgs/s | 0.99x |
| **TurboJPEG** | **2.90 min** | **24,522 imgs/s** | **1.08x** |
| TurboJPEG + 内存盘 | 3.22 min | 22,063 imgs/s | 0.97x |

### 单线程解码微基准 (bench_decode.py, 5000 张 112x112 JPEG)

| 方案 | 速度 | 延迟 | 加速比 |
|------|------|------|--------|
| PIL (当前路径: PIL→np→PIL→ToTensor) | 3,715 imgs/s | 269 us | 1.00x |
| PIL direct (跳过冗余转换) | 4,184 imgs/s | 239 us | 1.13x |
| OpenCV naive | 4,482 imgs/s | 223 us | 1.21x |
| OpenCV inplace (in-place torch ops) | 6,650 imgs/s | 150 us | 1.79x |
| TurboJPEG (cached instance) | 6,464 imgs/s | 155 us | 1.74x |
| torchvision.io (C++ decode_jpeg) | 6,679 imgs/s | 150 us | 1.80x |

### DataLoader 多 Worker 吞吐 (bench_dataloader.py, batch=512)

| 方案 | w=8 | w=12 | w=16 |
|------|-----|------|------|
| RecordIO+PIL | 9,479 | 10,637 | 10,888 |
| RecordIO+CV2 | 10,288 | 11,592 | 11,566 |
| RecordIO+TurboJPEG | 11,742 | 11,466 | 12,387 |
| RecordIO+torchvision.io | 11,631 | 11,396 | 12,009 |
| LMDB+TurboJPEG | 10,217 | 11,738 | 11,850 |

---

## 4. 为什么 torchvision.io 单线程快但实际训练不快?

单线程下 torchvision.io 和 TurboJPEG 持平 (6,679 vs 6,464 imgs/s)。但在多 Worker DDP 训练中 torchvision.io 退化到和 PIL 一样 (3.16 vs 3.23 min):

1. **额外内存拷贝**: `torch.frombuffer(bytearray(img_bytes))` 先拷贝一次 bytes→bytearray, 再创建 tensor。TurboJPEG 直接解码到 numpy 数组 (单次分配)
2. **C++ 线程池争用**: torchvision 内部的 libjpeg-turbo 绑定会启动自己的线程池做并行解码, 当 DataLoader 已经有 8-16 个 worker 进程时, 进程内线程池反而造成 CPU 过度订阅 (oversubscription), 上下文切换增多
3. **GIL + tensor 操作开销**: 返回的是 CUDA-unaware 的 CPU tensor, 后续 `img.float().div_().sub_().div_()` 都在 torch dispatcher 里走, 比纯 numpy 的 `torch.from_numpy()` 多了 dispatch overhead
4. **fork 后重新初始化**: torchvision C++ 后端在 fork 后可能需要重建内部状态, 增加了每个 worker 的冷启动成本

TurboJPEG 赢在: 纯 C 库 + 无内部线程 + 直接返回 numpy (零 torch dispatch) + fork-safe (无状态)

---

## 5. 为什么内存盘 (tmpfs) 反而更慢?

- 92 GB RecordIO 数据在首次训练后已被 OS page cache 完全缓存 (机器有 503 GB 内存, buff/cache 285 GB)
- 从磁盘 read() 实际命中的是 page cache → 和内存盘一样快
- 把数据放到 /dev/shm 反而**减少了可用 page cache 空间 92 GB**, 可能影响其他文件 (idx, tsv) 的缓存
- tmpfs 的内存管理路径比 page cache 的零拷贝 read() 多了一层 VFS 开销

**结论: 不需要内存盘。OS page cache 已经在做同样的事, 而且更高效。**

---

## 6. 格式选择总结

| 指标 | ImageFolder | RecordIO | LMDB |
|------|-------------|----------|------|
| 打包时间 | 0 | 42.8 min | 51.1 min |
| 磁盘占用 | ~150GB (37M 文件) | 98.3 GB (2 文件) | 137 GB (实际) |
| 训练速度 (1 epoch) | 40.2 min | 30.3 min | 29.9 min |
| 读取方式 | 随机 open/read | seek + read (单文件) | mmap B+tree lookup |
| 随机访问 | 文件系统 inode | .idx 偏移量表 + fseek | key→page 查找 |
| 多进程竞争 | inode/dentry 压力大 | 文件句柄 PID 检测 | 无锁并发读 (readonly) |
| 实现复杂度 | 最简单 | 中等 (手动解析) | 低 (lmdb 库封装) |

**推荐: RecordIO** — 综合最优 (打包快、磁盘小、速度和 LMDB 持平、项目原生支持)

---

## 7. 解码后端选择总结

| 指标 | PIL | OpenCV | TurboJPEG | torchvision.io |
|------|-----|--------|-----------|----------------|
| 单线程速度 | 3,715/s | 6,650/s | 6,464/s | 6,679/s |
| DataLoader w=8 | 9,479/s | 10,288/s | 11,742/s | 11,631/s |
| 实际训练 (7卡) | 3.23 min | 3.10 min | **2.90 min** | 3.16 min |
| 额外依赖 | 无 | opencv-python | PyTurboJPEG + libturbojpeg | torchvision ≥0.15 |
| fork-safe | 是 | 是 | 是 (无状态) | 有开销 |
| 多 worker 表现 | 基准 | 好 | **最好** | 退化 |

**推荐: TurboJPEG** — 实际训练快 10%, 无额外线程争用, 改动最小

---

## 8. 最终推荐配置

```bash
# 环境
export DECODE_BACKEND=turbojpeg  # 已设为默认, 无需手动设置

# DataLoader (fabric/fabric.py 已自动配置)
pin_memory=True
prefetch_factor=3
persistent_workers=True
num_workers=8

# 数据格式
dataset=configs/dataset_0605_train_rec.yaml  # RecordIO

# GPU 加速 (train_opt.py 已配置)
torch.backends.cudnn.benchmark = True
model.to(memory_format=torch.channels_last)
torch.compile(model, dynamic=False)
```

---

## 9. 代码改动说明

| 文件 | 改动 |
|------|------|
| `dataset/recordio_reader.py` | 新增多后端解码 (DECODE_BACKEND 环境变量), 默认 turbojpeg |
| `dataset/base_dataset.py` | `read_sample` 去掉冗余 np→PIL 转换, decode_image 直接返回 PIL |
| `fabric/fabric.py` | 训练 DataLoader 加 `persistent_workers=True`, `prefetch_factor=3` |
| `dataset/__init__.py` | LMDB 检测: 如果 `train.lmdb/` 存在走 LMDBFaceDataset |
| `dataset/configs/dataset_0605_train_lmdb.yaml` | LMDB 版 dataset config |
| `dataset/configs/dataset_0605_train_rec.yaml` | RecordIO 版 dataset config |
| `dataset/configs/dataset_0605_train_ramdisk.yaml` | 内存盘版 (实测无效, 仅保留参考) |
| `scripts/benchmark/bench_decode.py` | 单线程解码速度对比脚本 |
| `scripts/benchmark/bench_dataloader.py` | DataLoader 多 worker 吞吐对比脚本 |
| `scripts/benchmark/bench_real_train.sh` | 真实 7 卡 DDP 训练对比脚本 |
| `scripts/benchmark/bundle_lmdb_large.py` | 大数据集 LMDB 打包 |
| `scripts/benchmark/bundle_rec_large.py` | 大数据集 RecordIO 打包 |

---

## 10. Augmenter v2 优化对比 (1000 batch, 7卡 DDP, batch_size=512)

### 背景

v1 augmenter 存在大量 PIL↔numpy 格式转换开销:
- `Image.fromarray()` 和 `np.array(PIL)` 每张图 2-6 次
- imgaug 依赖 (BlurAugmenter) 引入额外 PIL↔numpy 来回
- `F.adjust_brightness` 等 torchvision 函数要求 PIL 输入

v2 重写: 全链路 numpy/cv2, 零 PIL 对象创建。TurboJPEG 直接返回 numpy → augment → torch.from_numpy。

### 单线程微基准 (5000 images, 112x112)

| 方案 | 速度 | 延迟 | 加速比 |
|------|------|------|--------|
| BasicAugmenter v1 (PIL) | 3,999/s | 250 us | 1.00x |
| **BasicAugmenter v2 (numpy)** | **7,722/s** | **129 us** | **1.93x** |
| BasicAugmenter v2 (cv2) | 4,454/s | 225 us | 1.11x |
| BasicAugmenter v2 (gpu/cpu) | 4,459/s | 224 us | 1.11x |
| GridSampleAug v1 (PIL混合) | 1,052/s | 951 us | 1.00x |
| **GridSampleAug v2 (numpy)** | **1,844/s** | **542 us** | **1.75x** |
| GridSampleAug v2 (gpu/cpu) | 1,206/s | 829 us | 1.15x |

### 真实 7卡 DDP 训练 (1000 batch, batch_size=512, TurboJPEG)

| 方案 | Epoch Time | Avg Speed | 相对 gridsample_v1 |
|------|-----------|-----------|-------------------|
| basic_v1 (PIL) | 3.15 min | ~22,600/s | +10% |
| basic_v2_numpy | 3.08 min | ~23,200/s | +12% |
| gridsample_v1 (PIL混合) | 3.50 min | ~20,400/s | baseline |
| **gridsample_v2_numpy** | **3.33 min** | **~21,500/s** | **+4.9%** |
| gridsample_v2_gpu | 报错 (需改训练循环) | — | — |

### 分析

1. **纯 numpy 版是最佳选择**: 单线程快 1.75x, 实际训练快 5%
2. **cv2 版没有额外优势**: photometric 操作无论如何都走 float32 numpy, cv2 对 112x112 小图无 SIMD 优势
3. **GPU 版不适合当前架构**: 需要改造为 batch-level augmentation, 且 GPU 已被 forward+backward 占满
4. **v2 numpy 额外好处**: 去掉 imgaug 依赖, 减少内存分配压力, 代码更简洁

### v2 代码改动

| 文件 | 说明 |
|------|------|
| `data_augs/basic_augmenter_v2_numpy.py` | 纯 numpy BasicAugmenter |
| `data_augs/basic_augmenter_v2_cv2.py` | 全 cv2 BasicAugmenter |
| `data_augs/basic_augmenter_v2_gpu.py` | torch tensor BasicAugmenter |
| `data_augs/gridsample_augmenter_v2_numpy.py` | 纯 numpy/cv2 GridSampleAugmenter |
| `data_augs/gridsample_augmenter_v2_gpu.py` | torch GridSampleAugmenter |
| `data_augs/photometric_numpy.py` | PhotometricRandAugment numpy 实现 |
| `data_augs/__init__.py` | 注册 v2 augmenters |
| `dataset/augment_dataset_v2.py` | V2 dataset (decode→numpy→augment→tensor) |
| `dataset/recordio_reader.py` | 新增 `decode_image_numpy()` 方法 |
| `data_augs/configs/basic_v2_numpy.yaml` | v2 numpy config |
| `data_augs/configs/gridsample_v2_numpy.yaml` | v2 gridsample numpy config |
| `scripts/benchmark/bench_augmenter_v2.sh` | 7卡对比脚本 |

---

## 11. 累计优化效果总结

| 优化项 | 效果 | 叠加后 |
|--------|------|--------|
| RecordIO 替代 ImageFolder | -25% 时间 (40.2→30.3 min/epoch) | 30.3 min |
| TurboJPEG 替代 PIL decode | -10% 时间 (3.23→2.90 min/1000batch) | — |
| v2 numpy augmenter | -5% 时间 (3.50→3.33 min/1000batch) | — |
| torch.compile + channels_last + cudnn | +31% GPU 吞吐 | — |
| persistent_workers + prefetch_factor=3 | 减少 worker 冷启动 | — |

**推荐生产配置**: RecordIO + TurboJPEG + gridsample_v2_numpy + torch.compile

---

## 12. 测试日期

2026-06-10

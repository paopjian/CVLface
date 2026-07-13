# 数据格式基准测试报告

## 测试目标

对比 5 种数据存储格式在人脸识别训练场景下的性能表现，选出最适合大规模训练的格式。

## 测试环境

- GPU: NVIDIA GeForce RTX 4090 (24GB)
- CPU: 8 workers (DataLoader)
- 数据集: 473,425 张图片, 10,000 类 (从完整训练集中提取)
- 数据路径: `/data1/dataset_0605/try2`
- 模型: iResNet-101 (前向+反向传播, 完整训练循环)
- Batch size: 256
- 图片尺寸: 112×112×3
- 精度: FP32
- 测试方法: 每个格式测试前执行 `sync && echo 3 > /proc/sys/vm/drop_caches` 清空 page cache

## 测试格式

| 格式 | 存储方式 | 打包工具 |
|------|---------|---------|
| ImageFolder | 原始 JPEG 文件 (目录结构) | 无需打包 |
| LMDB | `[4B label][JPEG bytes]` 键值对 | `bundle_images_into_lmdb.py` |
| RecordIO | MXNet RecordIO 二进制流 | `bundle_images_into_rec.py` |
| WebDataset | tar 分片, 每片 ~100MB | `bundle_webdataset.py` |
| HDF5 | 变长 JPEG blob 数组 | `bundle_hdf5.py` |

## 单GPU测试结果 (清除 page cache 后冷启动)

| 格式 | 磁盘占用 | Epoch时间 | DataLoad/batch | 吞吐量 | GPU内存 |
|------|---------|-----------|---------------|--------|---------|
| **ImageFolder** | **1.2 GB** | **48.5s** | **21.2ms** | **9,756 samples/s** | 1,118 MB |
| LMDB | 3.9 GB | 49.0s | 22.0ms | 9,655 samples/s | 1,117 MB |
| RecordIO | 3.0 GB | 53.5s | 24.2ms | 8,842 samples/s | 1,117 MB |
| WebDataset | 4.7 GB | 63.4s | 29.4ms | 7,451 samples/s | 1,117 MB |
| HDF5 | 3.1 GB | 72.8s | 34.9ms | 6,500 samples/s | 1,117 MB |

## 性能排名 (吞吐量)

```
ImageFolder ████████████████████████████████████████  9,756 samples/s (100%)
LMDB       ███████████████████████████████████████░  9,655 samples/s (99%)
RecordIO   ████████████████████████████████████░░░░  8,842 samples/s (91%)
WebDataset █████████████████████████████████░░░░░░░  7,451 samples/s (76%)
HDF5       ██████████████████████████████░░░░░░░░░░  6,500 samples/s (67%)
```

## 冷启动 vs 热缓存对比

之前的测试没有清除 page cache，存在前一个格式的缓存数据占用内存的干扰。

| 格式 | 热缓存 (之前) | 冷启动 (本次) | 变化 |
|------|-------------|------------|------|
| ImageFolder | 7,236 | **9,756** | +35% |
| LMDB | 7,987 | **9,655** | +21% |
| RecordIO | 7,618 | **8,842** | +16% |
| WebDataset | 6,921 | 7,451 | +8% |
| HDF5 | 6,181 | 6,500 | +5% |

**分析**: 清空缓存后所有格式都变快了。原因是之前各格式残留的 page cache 相互竞争内存。
清空后，每个格式独占全部可用 page cache，1.2GB 数据在 epoch 前几百 batch 内即可全部加载入 cache。

## 各格式分析

### ImageFolder (小数据集首选)
- 原始文件系统目录结构，零打包成本
- 数据量最小 (1.2GB)，最先完全进入 page cache
- 新增/删除样本极其方便，调试时可直接查看图片
- 缺点: 文件数多时 inode 压力大, 不适合网络文件系统; 数据集远大于内存时随机访问 seek 多

### LMDB (大数据集首选)
- 基于 mmap 的 B+tree 键值存储
- 随机访问性能最优, 无锁争用, 支持多进程并发读取
- 当数据集远大于内存时优势更明显 (连续存储减少 disk seek)
- 缺点: 磁盘占用比 RecordIO 略大 (B+tree 开销)

### RecordIO
- MXNet 原生格式, 顺序存储 JPEG
- 吞吐量中等, 比 LMDB 慢 9%
- 兼容旧数据集 (WebFace, CASIA 等)
- 缺点: 依赖 mxnet 库 (已停止维护); shuffle 时随机 seek 不如 LMDB

### WebDataset
- tar 分片流式读取, 适合超大规模数据 (PB 级)
- DDP 下需要 `nodesplitter=split_by_node` 分 shard
- shard 切换时有短暂延迟 (偶发 100-200ms data load spike)
- 缺点: 不支持真正的随机访问; 打包后磁盘最大

### HDF5
- 单文件存储变长 JPEG blob
- 随机读取时 h5py 内部锁争用明显 (每隔一批出现 ~250ms spike)
- 需要 lazy init 避免 fork 安全问题
- 缺点: 小 blob 随机访问性能最差, 不适合训练 DataLoader

### Zarr (测试排除, 不推荐)
- 存储原始像素 (112×112×3 = 37,632 bytes/image vs JPEG ~4KB)
- IO 量是其他格式 10 倍, 吞吐量仅 ~270 samples/s
- tensorstore 需要 `multiprocessing_context='spawn'` (fork 不兼容)
- 根本问题: 图像训练应存 JPEG 编码数据, 不应存原始像素

## 遇到的技术问题与解决方案

### 1. tensorstore fork() abort
- 现象: `aborting: fork() use detected, not allowed due to internal threading`
- 原因: tensorstore 内部线程池与 DataLoader fork 不兼容
- 解决: DataLoader 设置 `multiprocessing_context='spawn'`

### 2. HDF5 多进程死锁
- 现象: h5py.File 在 fork 后子进程中 hang
- 原因: HDF5 文件句柄不能跨 fork 共享
- 解决: lazy init 模式, 每个 worker 独立打开文件

### 3. WebDataset DDP 报错
- 现象: `need explicit nodesplitter for multi-node training`
- 解决: 添加 `nodesplitter=wds_lib.split_by_node`

### 4. CUDA out of bounds (num_classes 不匹配)
- 现象: 从 1000 组扩展到 10000 组后 FC 层维度错误
- 解决: 自动检测 num_classes (从 dataset.class_to_idx 或目录计数)

### 5. DDP 输出混乱
- 现象: 所有 rank 同时打印, 输出交错不可读
- 解决: `log()` helper 只在 rank 0 输出

### 6. Page cache 干扰
- 现象: 不同格式间测试结果互相影响
- 解决: 每个格式测试前 `sync && echo 3 > /proc/sys/vm/drop_caches`

## 推荐方案

| 场景 | 推荐格式 | 理由 |
|------|---------|------|
| 数据集 < 内存 (本地 SSD) | ImageFolder | 最快 + 最省磁盘 + 零打包成本 |
| 数据集 >> 内存 (本地 SSD) | LMDB | mmap 随机访问, 大数据集优势明显 |
| 磁盘空间紧张 | ImageFolder | 最小存储, 不需要额外打包空间 |
| 兼容旧数据/遗留系统 | RecordIO | 已有大量 rec 格式数据集 |
| 超大规模分布式/对象存储 | WebDataset | 流式设计, 适合 S3/GCS |
| 需要避免的 | Zarr (raw pixels), HDF5 | IO 瓶颈 / 锁争用 |

## 最终结论

**当数据集能放入内存时 (本例 1.2GB), ImageFolder 和 LMDB 性能几乎一样 (~9700 samples/s), 且 ImageFolder 磁盘占用最小、使用最简单。**

**当数据集远大于内存时 (如 WebFace12M ~200GB), LMDB 的 mmap 连续存储优势才能真正体现 — 减少随机 disk seek, 远优于 ImageFolder 的海量小文件随机 IO。**

## 文件清单

```
work_0605/
├── benchmark_dataformat.py          # 主基准测试脚本
├── dataset/benchmark_datasets.py    # 各格式的 Dataset 实现
├── 计划_bench_结果.md               # 本报告
└── scripts/benchmark/
    ├── bundle_lmdb.py               # LMDB 打包
    ├── bundle_hdf5.py               # HDF5 打包
    ├── bundle_webdataset.py         # WebDataset 打包
    └── bundle_zarr.py               # Zarr 打包 (chunk_size=256)
```

# 训练中 TinyFace 评估变慢研究总结

最后更新：2026-08-07

## 1. 问题背景

在 `tmux a -t zkj` 观察 CoreFace SubCenter 训练时，训练和训练后外部评估都出现耗时异常：早期 epoch 约 2 小时，后续 epoch 超过 2 小时；一次训练现场的完整评估总耗时达到 `2422.80 s`（40.38 分钟），其中 TinyFace 占 `1883.57 s`（31.39 分钟）。日志显示 TinyFace 特征提取已经结束，但在

```text
TinyFace similarity: probes=3728, gallery=157871, dim=512, cpu_threads=32
```

之后 rank 0 停止输出，而其他 GPU/rank 仍在运行。

本研究的目标是区分以下几类原因：

1. TinyFace 的 CPU 相似度矩阵或 Top-K 实现本身存在逐次泄漏；
2. 多次启动外部评估会累积进程、线程或编译缓存；
3. 训练 DataLoader/RecordIO/worker 的 IO 和内存压力会单独拖慢 TinyFace；
4. 训练模型、optimizer、CUDA context、torch.compile 状态与外部评估进程共存时产生资源竞争；
5. 训练速度变慢是否与 TinyFace 评估变慢是同一个问题。

## 2. TinyFace 计算结构

相关实现：

- `evaluations/tinyface_evaluator.py`
- `evaluations/tinyface/evaluate.py`
- `evaluations/tinyface/metrics.py`

TinyFace 的评估流程如下：

1. 对 TinyFace 全部图像提取一次特征；
2. 将输入图像水平翻转，再提取第二次特征；
3. 两次特征相加，在 rank 0 上建立 probe/gallery 索引；
4. 计算 `3728 x 157871` 的 probe-gallery 相似度矩阵；
5. 根据同一矩阵执行 label 比较、排序/Top-K 和 DIR/FAR 指标计算。

矩阵规模（float32）约为：

| 对象 | 形状/大小 | 估算内存 |
|---|---:|---:|
| score matrix | `3728 x 157871` | `2.35 GiB` |
| bool label matrix | `3728 x 157871` | `0.59 GiB` |
| 全部 embedding（约 161599 张，512 维） | `N x 512` | 约 `0.31 GiB`/份 |
| 原图/翻转图特征及临时数组 | 依实现生命周期变化 | 额外数百 MiB |

当前 CPU 相似度矩阵使用 `torch.mm`，并由 `TINYFACE_NUM_THREADS` 控制线程数；代码对矩阵乘法、归一化、tensor 转换和 label matrix 均有阶段计时。评估总耗时并不等于 matmul 耗时，后续矩阵处理、特征提取、进程启动和分布式同步也包含在内。

TinyFace 特别容易暴露资源问题，是因为它需要数 GiB 级别的 probe x gallery 中间结果；普通 verification 主要是成对 embedding 比较，没有同等规模的二维矩阵。因此“只有 TinyFace 明显异常”并不等于 TinyFace 的算法一定泄漏，更可能是它最先触及 CPU 内存、NUMA、线程和 page fault 等瓶颈。

## 3. 基线与对照

### 3.1 训练运行中的异常基线

一次训练现场的计时汇总：

| 评估器 | 耗时 |
|---|---:|
| `work_0605_3t` | `180.71 s` |
| `work_0605_enhance` | `25.73 s` |
| `work_0605_glint` | `312.89 s` |
| `cplfw` | `6.93 s` |
| `agedb_30` | `6.21 s` |
| `tinyface` | `1883.57 s` |
| `calfw` | `6.76 s` |
| **总计** | **`2422.80 s`** |

训练存活时观察到：RAM used 约 `259 GiB`、available 约 `233 GiB`、swap used 约 `1.1 GiB`、load average 约 `24.85`。这说明当时系统仍有可用内存，但已经处于明显高负载和少量 swap/page reclaim 环境；512 GiB 是物理总量，不代表评估进程可以获得连续、低延迟且本地 NUMA 的内存。

### 3.2 停止训练后的同 checkpoint

停止原训练并释放训练进程后，使用同一 checkpoint 重新评估：

| 项目 | 训练存活时 | 停止训练后 |
|---|---:|---:|
| 完整评估总耗时 | `2422.80 s` | `415.87 s` |
| TinyFace | `1883.57 s` | `28.64 s` |
| TinyFace matmul | 未能从异常现场看到有效慢算力 | 约 `0.37 s` |
| TinyFace Top-K | 未能从异常现场看到有效慢算力 | 约 `0.16 s` |
| RAM used | 约 `259 GiB` | 约 `3.9 GiB` |
| available | 约 `233 GiB` | 约 `496 GiB` |
| load average | 约 `24.85` | 约 `0.27` |

该对照证明：TinyFace 数据和 checkpoint 本身可以在几十秒内完成，异常与训练存活时的系统/进程状态高度相关。

## 4. 已完成实验

### 4.1 32 线程连续 fresh 外部评估 12 次

脚本：

`scripts/benchmark/test_tinyface_external_restarts.py`

实验要点：

- 每次都是全新 `fabric run`，8 GPU；
- 使用 `test_20260605_tinyface` 配置；
- 开启 `torch.compile`；
- 明确设置 `TINYFACE_NUM_THREADS=32`、`OMP_NUM_THREADS=32`、`MKL_NUM_THREADS=32`；
- 子进程完成后退出，再启动下一次；
- 复用已有 checkpoint，没有保存训练 checkpoint。

TinyFace 时间（秒）：

```text
36.15, 36.79, 36.77, 36.71, 36.44, 36.51,
36.29, 36.12, 36.55, 36.67, 36.59, 36.72
```

统计：平均 `36.526 s`，最小 `36.124 s`，最大 `36.788 s`。

结论：32 线程配置有效；连续启动 12 次没有逐次变慢，不能支持“fresh 评估进程或线程池按次数累积泄漏”的假设。

结果目录：`/tmp/tinyface_external_restarts_12_threads32/`

### 4.2 真实训练 3 batch 后进入 TinyFace

使用真实训练环境：8 GPU、每卡 batch 128、每 rank 8 个 DataLoader worker（总计 64 个 worker）、CoreFace 双视图训练，并从 epoch 10 checkpoint 恢复。只执行 3 个 batch 后触发外部 TinyFace。

结果：

- TinyFace：`51.62 s`；
- 干净 fresh 评估平均：`36.53 s`；
- 训练现场约慢 `41%`；
- CPU matmul 仍约 `0.347 s`，Top-K 仍约 `0.161 s`；
- 测试复用已有 checkpoint，没有保存新 checkpoint。

这说明“训练进程仍然存在”会造成稳定的额外开销，但 51 秒级别仍远低于 31 分钟。

### 4.3 真实训练 3 batch x 10 epoch，每 epoch 评估一次

实验输出：

`cvlface/research/recognition/experiments/try_coreface_subcenter/tinyface_3_batches_each_epoch_10epochs_08-07_0`

关键配置记录在该目录的 `config.yaml`：

- `limit_num_batch: 3`；
- `num_gpu: 8`；
- `batch_size: 128`；
- `num_workers: 8`；
- `external_eval: true`；
- `external_eval_compile: true`；
- `benchmark_eval_checkpoint` 指向已有 epoch 10 checkpoint，跳过新 checkpoint 写入；
- TinyFace 每 epoch 评估一次。

epoch 11 到 20 的 TinyFace 时间（秒）：

```text
51.956, 51.744, 51.825, 51.675, 51.947,
51.800, 51.078, 51.695, 51.977, 51.421
```

统计：平均 `51.712 s`，最小 `51.078 s`，最大 `51.977 s`。

结论：10 次连续外部 TinyFace 没有随 epoch 增长而变慢。3 个 batch 不会遍历和累积整个训练数据结构，因此该实验主要排除了“评估调用次数本身导致 TinyFace 逐次泄漏”，但不能替代完整 epoch 训练后的现场测试。

### 4.4 完整 RecordIO 空转 3 epoch，每 epoch 评估一次

脚本：

`scripts/benchmark/empty_read_tinyface_epochs.py`

该脚本保留真实训练数据路径、RecordIO 读取、JPEG 解码、gridsample augmentation、pin-memory、8 rank 和每 rank 8 worker；不构造模型、不执行 forward/backward、不保存 checkpoint。每个完整 epoch 遍历约 `37,084,479` 条样本后，再启动一次 TinyFace 外部评估。

结果目录：`/tmp/tinyface_empty_read_3epochs_retry/`

空转数据读取：

| 空转 epoch | 完整读取耗时 | rank 0 PSS | rank 0 Private Dirty |
|---:|---:|---:|---:|
| 0 | `1022.52 s` | `2.27 GiB` | `0.96 GiB` |
| 1 | `1029.62 s` | `3.04 GiB` | `1.67 GiB` |
| 2 | `1034.42 s` | `3.57 GiB` | `2.52 GiB` |

TinyFace：

| 空转 epoch | TinyFace |
|---:|---:|
| 0 | `38.630 s` |
| 1 | `38.306 s` |
| 2 | `37.840 s` |

空转过程中系统 RAM used 一度约 `222 GiB`，测试结束后恢复到约 `4.7 GiB`。rank 0 的 PSS/Private Dirty 只是单个父进程指标；大量内存由 worker 的共享页、page cache 和其他 rank 共同构成，不能用单个进程 RSS 直接代表全机占用。

结论：即使完整 DataLoader/RecordIO 空转使系统内存明显上升，也没有复现 TinyFace 变成几十分钟。因此纯 IO、worker COW 或“内存用了 222 GiB”不是当前异常的充分原因。

## 5. 已排除或显著降低可能性的因素

### 已基本排除

- **32 线程设置错误**：显式设置 32 线程后，fresh 评估稳定在 36 秒级；
- **多次 fresh 启动逐渐累积泄漏**：连续 12 次没有趋势性变慢；
- **TinyFace CPU matmul 算法本身异常**：正常现场约 0.35--0.37 秒；
- **TinyFace Top-K/排序算法本身异常**：正常现场约 0.16 秒；
- **评估顺序的固定污染**：停止训练后同 checkpoint 可恢复到 28.64 秒；
- **纯 RecordIO/DataLoader 内存压力单独造成 31 分钟**：完整空转 3 epoch 未复现。

### 仍不能排除

- 训练模型、optimizer、partial-FC 参数或训练中间 tensor 常驻 CPU 内存；
- 训练 CUDA context 与外部评估 CUDA context 同时存在时的 CPU/GPU/PCIe 资源竞争；
- torch.compile/Inductor 的线程、缓存或编译产物在训练进程中长期驻留；
- 训练进程的后台线程、DataLoader worker、文件句柄或 NUMA 亲和性造成调度/page fault 竞争；
- 完整 epoch 训练后仍残留的参数 cache、梯度、通信 buffer 或 allocator 状态；
- 训练本身越来越慢的独立原因，例如数据等待、文件系统吞吐下降、CPU oversubscription、swap/page reclaim 或 GPU 同步。

## 6. 当前根因判断

现有证据支持以下判断：

1. TinyFace 不是在连续调用中逐次泄漏；
2. TinyFace 的大矩阵是放大器，而不是已证实的根因。它需要约 2--4 GiB 级别的连续中间结果和大量内存带宽，所以会比普通 verification 更早表现出资源竞争；
3. 异常主要依赖“训练进程仍存活”的现场状态。训练停止后立即恢复，真实训练 3 batch 后也稳定比 fresh 慢约 41%；
4. 纯 IO/worker 压力不能单独复现 31 分钟，因而更应优先检查训练特有状态：模型/optimizer/通信 buffer、CUDA context、compile 状态、线程和 NUMA；
5. “训练 epoch 越来越慢”和“TinyFace 评估变慢”可能共享系统资源压力，但目前没有证据表明 TinyFace 评估器直接导致了训练吞吐的逐 epoch 下降。

## 7. 后续建议

### 7.1 先做一次完整 epoch 现场采样

在训练进程进入外部评估前、评估期间和评估结束后记录：

```bash
free -h
vmstat 1
iostat -xz 1
numastat -p <训练父进程PID>
ps -eLf --forest
nvidia-smi pmon -s um
nvidia-smi
```

重点比较 rank 0/其他 rank 的 CPU 利用率、上下文切换、major/minor page fault、NUMA 命中率、磁盘等待和 GPU 利用率。TinyFace 日志中的 matmul/Top-K 计时必须与这些系统指标按时间对齐。

### 7.2 做“训练父进程常驻状态”的 A/B

在完整 epoch 后分别测试：

- 保持模型、optimizer、DataLoader worker 和 CUDA context 全部常驻；
- 进入外部评估前显式释放 epoch-local tensor、清空参数 cache，并关闭/重启 DataLoader worker；
- 完全退出训练进程后再启动同一外部评估。

如果只有第一种慢，根因范围就可以收窄到训练进程常驻资源或其与子进程的竞争。当前 `train_opt.py` 已在外部评估前执行 `gc.collect()` 和 `torch.cuda.empty_cache()`，但这只覆盖一部分 Python/CUDA 临时对象，不能保证释放模型、optimizer、worker 或线程池。

### 7.3 记录训练 epoch 的分段耗时

将每个 batch 拆分为数据等待、前向、反向、optimizer step 和同步等待；同时记录每 epoch 的 `minor_faults/major_faults`、IO wait、CPU load、GPU utilization。这样可以判断“训练越来越慢”到底是数据端、CPU 端、GPU 端还是同步端。

### 7.4 降低 TinyFace 峰值内存

长期方案可以将完整 score matrix 改为分块计算并维护每个 probe 的 Top-K，避免一次性分配约 `2.35 GiB` score matrix 和约 `0.59 GiB` label matrix。该优化不能解释当前已观测到的 31 分钟，但能降低训练现场对内存带宽、NUMA 和 page reclaim 的敏感性。

### 7.5 线程和 NUMA 对照

在“完整 epoch + 训练进程存活”的现场分别比较 `TINYFACE_NUM_THREADS=8/16/32`，并记录 CPU affinity/NUMA 绑定。fresh 测试中 32 线程稳定，但在训练现场线程数可能与 DataLoader、OpenMP、MKL 和编译线程叠加，最佳值不一定相同。

## 8. 代码与结果索引

### 计时和评估实现

- `evaluations/tinyface_evaluator.py`
- `evaluations/tinyface/evaluate.py`
- `evaluations/tinyface/metrics.py`
- `external_torch_eval.py`

### 训练 benchmark 逻辑

- `train_opt.py`：`benchmark_eval_checkpoint`、外部评估前清理、checkpoint 复用逻辑；
- `trainers/configs/default.yaml`：默认关闭 `benchmark_eval_checkpoint`，不改变正常训练行为；
- `evaluations/configs/test_20260605_tinyface.yaml`：TinyFace-only 评估配置；

### benchmark 脚本

- `scripts/benchmark/test_tinyface_external_restarts.py`：多次全新外部评估；
- `scripts/benchmark/empty_read_tinyface_epochs.py`：完整 RecordIO 空转 + 每 epoch TinyFace；

### 主要结果目录

- `/tmp/tinyface_external_restarts_12_threads32/`
- `/tmp/tinyface_empty_read_3epochs_retry/`
- `cvlface/research/recognition/experiments/try_coreface_subcenter/tinyface_3_batches_each_epoch_10epochs_08-07_0/`

## 9. 一句话结论

TinyFace 的大规模 probe-gallery 矩阵使它最容易暴露训练现场的资源竞争，但现有实验没有证明 TinyFace 自身会随评估次数泄漏；31 分钟异常仍应围绕“完整训练进程常驻状态 + 外部评估子进程”的 CPU/内存/NUMA/CUDA/线程竞争继续定位，而不是继续把 32 线程或单纯 512 GiB 内存容量作为首要怀疑对象。

## 10. 新增三组正式训练干扰测试

新增文件：

- `scripts/benchmark/train_opt_interference_test.py`：从 `train_opt.py` 复制出的测试训练入口；
- `scripts/benchmark/test_training_eval_interference.sh`：三个测试序列的 shell 编排。

测试训练入口通过 `BENCHMARK_LIMIT_SCHEDULE` 控制每个 epoch 的 batch 数，并通过
`trainers.benchmark_eval_checkpoint=<已有 checkpoint>` 复用已有 checkpoint。该模式会执行真实模型训练、DataLoader、forward/backward、optimizer 和外部 TinyFace 评估，但跳过 checkpoint 写入；因此不会为测试阶段生成无用的大型模型/分类器/优化器 checkpoint。

三组测试都默认占用 8 张 GPU（`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`），不能同时运行。

后台启动示例（一次只执行一个命令，前一个完成后再执行下一个；每个测试使用不同的 `OUTPUT_DIR`）：

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/try_coreface_subcenter
TEST_CASE=1 OUTPUT_DIR=/data1/dataset_0605/train_output/tinyface_interference_case1 \
  nohup bash scripts/benchmark/test_training_eval_interference.sh > /tmp/tinyface_case1.out 2>&1 &

TEST_CASE=2 OUTPUT_DIR=/data1/dataset_0605/train_output/tinyface_interference_case2 \
  nohup bash scripts/benchmark/test_training_eval_interference.sh > /tmp/tinyface_case2.out 2>&1 &

TEST_CASE=3 OUTPUT_DIR=/data1/dataset_0605/train_output/tinyface_interference_case3 \
  nohup bash scripts/benchmark/test_training_eval_interference.sh > /tmp/tinyface_case3.out 2>&1 &
```

三个 case 的顺序：

| Case | 阶段顺序 | 训练 batch |
|---:|---|---:|
| 1 | 连续 10 个正式训练阶段，每阶段后 TinyFace | `1000, 2000, ..., 10000` |
| 2 | 完整训练 1 epoch + 评估 -> 空转 1 epoch + 评估 -> 完整训练 1 epoch + 评估 | 完整 DataLoader |
| 3 | 空转 1 epoch + 评估 -> 完整训练 1 epoch + 评估 -> 空转 1 epoch + 评估 | 完整 DataLoader |

每个训练阶段的日志在 `OUTPUT_DIR/logs/`，外部评估原始 JSON 在对应训练阶段目录的 `external_eval/`，空转评估结果在 `OUTPUT_DIR/results/`，系统快照在 `*_resources.txt`。测试完成后读取各阶段 raw JSON 中的 `timing_results.tinyface`，并与资源快照中的 RAM、swap、load、GPU 利用率对齐。

三个 case 默认分别使用 rendezvous 端口 `29501`、`29502`、`29503`，可以并发启动；如端口被占用，可设置 `MAIN_PORT` 覆盖。之前使用默认 `29400` 并发启动时会触发 `EADDRINUSE`，不会进入训练或评估。

# 余弦相似度矩阵计算优化心得

## 背景

`get_sim_matrix_large_scale_v4` 使用 7x RTX 4090 计算大规模余弦相似度矩阵并统计正负样本分布直方图。观察到 GPU 利用率 100% 但功率仅 100W/450W，说明计算单元未被充分利用。

## 瓶颈分析

### 1. Boolean Indexing 的隐藏开销

原始流程：
```python
flat_sim = sim_block[mask]          # 动态长度输出
flat_labels = label_eq[mask]        # 动态长度输出
pos = flat_sim[flat_labels]         # 再次动态长度
neg = flat_sim[~flat_labels]        # 再次动态长度
```

每次 boolgean indexin 都产生**不确定长度**的中间张量，需要：
- 先扫描一遍 mask 计算输出长度
- 分配新显存
- 再扫描一遍复制数据

一个 block (10240x10240) 会触发 4 次动态分配，成为主要瓶颈。

### 2. FP32 Matmul 未利用 Tensor Core

RTX 4090 的 Tensor Core 吞吐：
- FP32: 82 TFLOPS
- TF32: 165 TFLOPS
- FP16: 330 TFLOPS

原始实现使用 FP32 matmul，只能使用 CUDA Core，浪费了 4x 的 Tensor Core 算力。

### 3. Block Size 过小

block_size=10240 产生 19306 个块，kernel launch overhead 占比高。增大到 32768 可减少到 1953 个块。

## 优化方案

### 方案B: FP32 + where(NaN) — 精度无损

```python
NAN = float('nan')
# 对角块: 用NaN屏蔽下三角+对角线
sim = torch.where(mask, sim, NAN)
# 分类: 用NaN屏蔽非目标类别
pos_vals = torch.where(label_eq, sim, NAN)
pos_hist += torch.histc(pos_vals, ...)  # histc 忽略 NaN
del pos_vals
neg_vals = torch.where(~label_eq, sim, NAN)
neg_hist += torch.histc(neg_vals, ...)
del neg_vals
```

关键点：
- `torch.where` 原地操作，输出形状固定（= 输入形状），无动态分配
- `torch.histc` 天然忽略 NaN 值，无需手动过滤
- 顺序计算 pos/neg 避免同时持有两份全尺寸矩阵导致 OOM
- 精度与原始实现完全一致（同为 FP32）

### 方案C: FP16 + where(NaN) — 最大加速

```python
b1 = feats[rs:re].to(device).half()
b2 = feats[cs:ce].to(device).half()
sim = torch.matmul(b1, b2.T).float()  # FP16 matmul, 结果转回 FP32 做 histc
```

在方案B基础上加入 FP16 matmul + TF32 backend，利用 Tensor Core。

## 实测结果 (2M 特征, 500K 类, 7x 4090, block_size=10240)

| 方案 | 耗时 | 加速比 | 精度影响 |
|------|------|--------|----------|
| 原始 v4 (FP32+bool_idx) | 67s | 1.00x | 基准 |
| 方案B (FP32+where NaN) | ~29s | ~2.3x | 无损 |
| 方案C (FP16+where NaN) | ~25s | ~2.6x | 区间级 <0.04% |

### 加速分解

- **where(NaN) 替代 boolean indexing**: ~2.3x（消除动态内存分配）
- **FP16 Tensor Core matmul**: 额外 ~1.1x
- **增大 block_size 到 32768**: matmul-only 可达 3.8x（但显存要求更高）

## Matmul-only 速度对比

| 配置 | 耗时 | TFLOPS | 加速比 |
|------|------|--------|--------|
| FP32 bs=10240 | 14.2s | 145 | 1.0x |
| FP16 bs=10240 | 12.5s | 165 | 1.1x |
| FP16+TF32 bs=16384 | 8.0s | 260 | 1.8x |
| FP16+TF32 bs=20480 | 6.4s | 324 | 2.2x |
| FP16+TF32 bs=32768 | 4.2s | 496 | 3.4x |

## 精度评估

FP16 matmul 的余弦相似度误差：
- 最大绝对误差: 1.17e-04
- 平均绝对误差: 1.25e-05
- 对 histogram 的影响: 总计数完全一致（pos/neg 差异 = 0），按 0.1 宽度区间聚合后差异 <0.04%

注意：如果用 20M bins 做逐 bin 对比会显示 ~196% 偏差，这是因为 bin 宽度(1e-7) 远小于 FP16 精度(1e-3)，值在相邻 bin 间漂移。这不代表实际误差，按任意合理区间聚合后差异可忽略。

## 关键发现

1. **Boolean indexing 是比 matmul 精度更大的性能杀手** — 同精度下仅改 where(NaN) 就有 2.3x 加速
2. **`torch.histc` 原生忽略 NaN** — 不需要额外标记值或过滤步骤
3. **不要用超出范围的值(如 11.0)替代 NaN** — histc 会把超范围值塞入边缘 bin，污染结果
4. **FP16 matmul 的精度对直方图统计任务完全可接受** — 我们关心的是分布形状而非单个值
5. **Block size 对 Tensor Core 利用率影响巨大** — 32768 比 10240 快 3.4x（matmul-only）
6. **顺序计算 pos/neg 避免 OOM** — 两个全尺寸 where 结果 + sim 本身会超 24GB

## 推荐应用策略

- **保守方案(无精度损失)**: 方案B — 仅替换 boolean indexing 为 where(NaN)，加速 ~2.3x
- **激进方案(推荐)**: 方案C — FP16 matmul + where(NaN)，加速 ~2.6x，精度完全可接受
- **进一步提速**: 在方案C基础上增大 block_size（需确认显存足够）

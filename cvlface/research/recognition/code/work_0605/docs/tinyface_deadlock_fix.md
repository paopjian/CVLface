# TinyFace 评估死锁修复

## 问题现象

8卡 DDP 训练中，epoch 0 的 tinyface 评估正常完成（~30s），但 epoch 1 进入评估后进程完全卡死，最终触发 NCCL 2小时超时：

```
mate probes: 3728, non mate probes: 0
[rank6]: WorkNCCL(...) ran for 7200015 milliseconds before timing out.
```

## 排查过程

1. 初始怀疑：barrier 同步逻辑问题 → 排除（所有 rank 正确到达 barrier）
2. 怀疑内存不足/swap → 排除（503GB RAM，172GB available，RSS 864MB 不增长）
3. 加诊断 print 定位 → epoch 1 卡在 `np.argsort` 之后的 `np.take_along_axis`
4. RSS 不增长说明进程没在计算，是 **死锁** 而非慢

## 根因

**numpy OpenMP/BLAS 线程池死锁**

PyTorch DDP 训练循环使用 ATen 并行后端（OpenMP/MKL），训练过程中对线程池状态的修改导致后续 numpy 的并行操作在获取 OpenMP 锁时死锁。epoch 0 线程池状态碰巧未损坏，epoch 1 训练完后状态坏了。

关键证据：
- 同样的数据和代码，epoch 0 正常，epoch 1 死锁
- `torch.argsort`（2.3s）正常完成后，紧接的 `np.take_along_axis` 死锁
- 进程 RSS 不增长 → 卡在锁等待，不是在运算

## 修复方案

将 tinyface 评估中的大规模数值运算从 numpy 改为 torch CPU 操作，避免触碰 numpy 的 OpenMP 线程池。

### 修改文件

**`evaluations/tinyface/evaluate.py`**

- `inner_product()` 原使用 `np.dot(x1, x2.T)` + `np.linalg.norm`
- 改为 `torch.mm` + `torch.norm`，命名为 `inner_product_torch()`

**`evaluations/tinyface/metrics.py`**

- `np.min(score_mat)` → `torch.from_numpy(score_mat).min().item()`
- `np.argsort(score_mat_m, axis=1)` + 逐行翻转 → `torch.argsort(..., descending=True)`
- `np.take_along_axis(label_mat, sort_idx)` → `torch.gather(...)`
- 去掉了 `threadpool_limits(limits=1)` hack

### 性能对比

| 操作 | 修改前 (numpy) | 修改后 (torch CPU) |
|------|---------------|-------------------|
| inner_product | 1.3s | 0.4s |
| argsort | 21.4s | 2.4s |
| 总计 | 29.6s | ~5s |

## 经验总结

在 PyTorch DDP 训练环境下，评估阶段的 numpy 大矩阵运算存在 OpenMP 线程池死锁风险。建议所有大规模数值运算（矩阵乘法、排序、归一化）使用 torch CPU 替代 numpy，因为 torch 的线程池和 DDP 训练共享同一个 ATen 并行后端，不会产生锁竞争。

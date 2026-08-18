# `run_rank_zero_metric` 方法分析与实现

## 问题背景

在 `qgface_subcenter` 中，`run_combined_evaluations_distributed` 函数调用了 `coordinator.run_rank_zero_metric()`，但这个方法在整个代码库中**从未被定义**。

## 用途说明

### 1. **为什么需要这个方法？**

`run_rank_zero_metric` 的作用是：**在分布式训练中，让 rank 0 执行需要多 GPU 的计算任务，同时让其他 rank 安全等待，避免 NCCL barrier 死锁**。

### 2. **具体使用场景**

在 `evaluations/__init__.py:129-141` 中：

```python
def run_combined_evaluations_distributed(
    fabric, evaluators_dict, combined_config, epoch, step, n_images_seen
):
    """Run combined GPU metrics without parking non-zero ranks in NCCL."""
    if not evaluators_dict:
        return {}
    coordinator = next(iter(evaluators_dict.values()))  # 取第一个 evaluator
    return coordinator.run_rank_zero_metric(
        lambda: run_combined_evaluations(evaluators_dict, combined_config),
        epoch,
        f"combined_{step}",
        n_images_seen,
    )
```

**调用链：**
1. `train_opt.py:649` → `run_combined_evaluations_distributed()`
2. → `coordinator.run_rank_zero_metric(compute_fn=lambda: run_combined_evaluations(...))`
3. → rank 0 执行 `run_combined_evaluations()` → 调用 `get_sim_matrix_large_scale_v4(num_gpus=8)`
4. → v4 内部启动 `ThreadPoolExecutor(max_workers=8)`，8 个 GPU worker 并行计算相似度矩阵

### 3. **为什么会死锁？**

**旧版本（qgface）的错误模式：**
```python
if fabric.local_rank == 0:
    result = run_combined_evaluations(...)  # 内部启动 8-GPU ThreadPoolExecutor
fabric.barrier()  # ❌ rank 1-7 在这里等待，但 GPU 1-7 正在被 rank 0 的线程池使用
```

**死锁原因：**
- **rank 0**: 主线程在 `ThreadPoolExecutor` 的 `future.result()` 上等待，等待 GPU 1-7 完成计算
- **rank 1-7**: 主线程在 `fabric.barrier()` 上等待 rank 0，但无法响应 rank 0 线程池的 GPU 调用
- **GPU 1-7**: 被 rank 0 的 worker 线程占用，但这些 GPU 的主进程（rank 1-7）在 NCCL barrier 中忙等
- **结果**: 循环依赖，30 分钟后 NCCL timeout

### 4. **`run_rank_zero_metric` 的设计目标**

使用**状态文件同步**替代 NCCL barrier，让非零 rank 轮询文件而非进入 NCCL 等待：

```
rank 0:  执行多 GPU 计算 → 写状态文件 "ok" → barrier
rank 1-7: 轮询状态文件（不占用 NCCL） → 读到 "ok" → barrier
```

这样 GPU 1-7 可以自由地被 rank 0 的线程池调用，不会与 rank 1-7 的主线程 NCCL 状态冲突。

---

## 完整实现

### 在 `BaseEvaluator` 或 `CustomVerificationEvaluator` 中添加：

```python
def run_rank_zero_metric(self, compute_fn, epoch, name_suffix, n_images_seen):
    """
    Execute compute_fn on rank 0 with status-file sync for other ranks.
    
    This method prevents NCCL deadlock when compute_fn uses multi-GPU workers
    (e.g., ThreadPoolExecutor with num_gpus=8). Non-zero ranks poll a status
    file instead of entering an NCCL barrier while rank 0's worker threads
    occupy their GPUs.
    
    Args:
        compute_fn: Callable that returns result dict, executed only on rank 0
        epoch: Current epoch number
        name_suffix: Unique suffix for the sync file (e.g., "combined_123")
        n_images_seen: Total images seen (for uniqueness)
    
    Returns:
        dict: Result from compute_fn (rank 0) or empty dict (other ranks)
    """
    import time
    
    # Generate unique sync file path
    metric_sync_path = self._metric_sync_path(epoch, name_suffix, n_images_seen)
    
    # Clean up old sync file before starting
    if self.fabric.local_rank == 0:
        try:
            os.remove(metric_sync_path)
        except FileNotFoundError:
            pass
    
    # Barrier before metric starts (all ranks ready)
    self.fabric.barrier()
    
    # Rank 0 executes multi-GPU metric, other ranks wait via file polling
    if self.fabric.local_rank == 0:
        try:
            start_time = time.time()
            result = compute_fn()
            elapsed = time.time() - start_time
            print(f"Rank 0 metric '{name_suffix}' completed in {elapsed:.2f}s")
            
            # Publish success status
            self._publish_metric_status(metric_sync_path, "ok")
        except Exception as error:
            # Publish error status to notify other ranks
            self._publish_metric_status(metric_sync_path, "error", error)
            raise
    else:
        result = {}
        # Poll status file instead of NCCL barrier
        self._wait_for_metric_status(metric_sync_path)
    
    # After metric completes, all ranks synchronize via NCCL
    self.fabric.barrier()
    torch.cuda.empty_cache()
    
    return result
```

### 辅助方法（已在 `CustomVerificationEvaluator` 中实现）

这些方法已经存在，可以直接复用：

- `_metric_sync_path(epoch, step, n_images_seen)` - 生成唯一状态文件路径
- `_publish_metric_status(path, status, error=None)` - rank 0 原子写入状态
- `_wait_for_metric_status(path)` - 非零 rank 轮询文件（默认超时 7200 秒）

如果要在 `BaseEvaluator` 中实现，需要将这三个辅助方法也移到基类。

---

## 实现位置建议

### 方案 1: 在 `CustomVerificationEvaluator` 中添加（推荐）

**优点:**
- 辅助方法已存在，直接添加即可
- `coordinator` 在实际使用中就是 `CustomVerificationEvaluator` 实例

**缺点:**
- 如果未来有其他 evaluator 类型需要此功能，需要重复实现

### 方案 2: 在 `BaseEvaluator` 中添加

**优点:**
- 所有 evaluator 子类都能继承
- 更符合面向对象设计

**缺点:**
- 需要将三个辅助方法也移到 `BaseEvaluator`
- `TinyFaceEvaluator` 等其他子类可能不需要此功能

---

## 验证清单

实现后需要验证：

- [ ] 使用 `combined_evaluations` 配置（如 `val_20260320.yaml`）启动训练
- [ ] 确认 rank 0 成功执行 `run_combined_evaluations`
- [ ] 确认 rank 1-7 在轮询状态文件期间 GPU 可被 rank 0 使用
- [ ] 确认状态文件路径唯一（包含 epoch/step/n_images_seen）
- [ ] 确认异常传播正确（rank 0 失败时其他 rank 也抛出异常）
- [ ] 确认没有 NCCL timeout（原 30 分钟死锁问题已解决）

---

## 与 `evaluate` 方法中的状态同步对比

| 特性 | `evaluate` 方法 | `run_rank_zero_metric` |
|------|----------------|------------------------|
| 触发条件 | `self.type == '4'` | 调用时总是启用 |
| 同步对象 | `compute_metric()` | 任意 `compute_fn` |
| 状态文件命名 | `{name}_{epoch}_{step}_{n_images_seen}` | `{name}_{epoch}_{name_suffix}_{n_images_seen}` |
| 返回值 | result dict | result dict |
| 错误处理 | 状态文件 + raise | 状态文件 + raise |

两者使用相同的底层机制（状态文件轮询），只是封装层次不同：
- `evaluate`: 针对单个 evaluator 的 `compute_metric` 阶段
- `run_rank_zero_metric`: 通用的 rank-0-only 计算包装器

---

## 总结

`run_rank_zero_metric` 是一个**通用的分布式同步包装器**，用于：

1. **让 rank 0 安全地执行多 GPU 计算**（如 ThreadPoolExecutor）
2. **让其他 rank 通过文件轮询而非 NCCL barrier 等待**
3. **避免"rank 0 等待其他 GPU，其他 rank 等待 rank 0"的循环死锁**

它是 `CustomVerificationEvaluator.evaluate` 中状态同步模式的泛化版本，专门用于 `combined_evaluations` 这类需要在 rank 0 启动多 GPU worker pool 的场景。

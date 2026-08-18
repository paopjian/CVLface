# qgface vs qgface_subcenter - Barrier 问题修复总结

**日期**: 2026-08-18  
**问题文档**: `cvlface/research/recognition/code/evaluation_barrier_issue.md`

---

## 核心问题

分布式训练中，当 rank 0 执行需要多 GPU 的 metric 计算时（如启动 8-GPU ThreadPoolExecutor），rank 1-7 若同时进入 NCCL barrier 等待，会导致**循环死锁**：

```
rank 0:   等待 GPU 1-7 完成计算（ThreadPoolExecutor worker）
rank 1-7: 等待 rank 0 完成（NCCL barrier）
GPU 1-7:  被 rank 0 的线程占用，但对应进程（rank 1-7）在 barrier 中忙等
→ 30 分钟后 NCCL timeout
```

---

## 修复方案：状态文件同步

**推荐模式**（来自 `evaluation_barrier_issue.md`）：

```python
sync_path = make_metric_sync_path(epoch, step, evaluator_name)

if fabric.local_rank == 0:
    remove_if_exists(sync_path)
fabric.barrier()  # ✅ 只在 metric 启动前同步

if fabric.local_rank == 0:
    try:
        result = compute_metric(...)
        publish_status_atomically(sync_path, status="ok")
    except Exception as error:
        publish_status_atomically(sync_path, status="error", error=error)
        raise
else:
    wait_for_status_file(sync_path)  # ✅ 轮询文件，不进入 NCCL
    result = {}

# ✅ metric 完成后才执行 barrier
fabric.barrier()
```

**关键点**：
- 非零 rank 通过**轮询文件**（CPU 操作）等待，不占用 NCCL 通信
- rank 0 的多 GPU worker 可以自由使用 GPU 1-7，不会与它们的主进程冲突

---

## 修复对比

### 1. CustomVerificationEvaluator (type='4' 评估器)

#### qgface（旧版）- ❌ 有死锁风险

```python
# qgface/evaluations/custom_verification_evaluator.py:620-635
if self.fabric.local_rank == 0:
    result = self.compute_metric(collection, collection_flip)  # 可能启动多 GPU
    self.log(result, epoch, step, n_images_seen)
    del collection, collection_flip
else:
    result = {}
    if not use_cache:
        del collection, collection_flip

# ❌ 直接 barrier，如果 compute_metric 使用多 GPU 会死锁
self.fabric.barrier()
torch.cuda.empty_cache()
return result
```

#### qgface_subcenter（新版）- ✅ 已修复

```python
# qgface_subcenter/evaluations/custom_verification_evaluator.py:640-695

# 新增辅助方法（595-631行）
def _metric_sync_path(self, epoch, step, n_images_seen):
    """生成唯一状态文件路径"""
    master_port = os.environ.get("MASTER_PORT", "default")
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "default")
    safe_name = self.name.replace(os.sep, "_").replace("/", "_")
    root = os.path.join("/tmp", "qgface_metric_sync", f"{master_port}_{run_id}")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{safe_name}_{epoch}_{step}_{n_images_seen}.json")

@staticmethod
def _publish_metric_status(path, status, error=None):
    """原子写入状态文件"""
    payload = {"status": status}
    if error is not None:
        payload["error"] = repr(error)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)  # 原子替换

@staticmethod
def _wait_for_metric_status(path):
    """轮询状态文件（默认超时 7200 秒）"""
    timeout = float(os.environ.get("QGFACE_METRIC_SYNC_TIMEOUT_SEC", "7200"))
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for metric status: {path}")
        time.sleep(0.1)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "ok":
        raise RuntimeError(f"Rank 0 metric failed: {payload.get('error', 'unknown error')}")

# evaluate 方法中的修复
metric_sync_path = self._metric_sync_path(epoch, step, n_images_seen) if self.type == '4' else None
if metric_sync_path is not None:
    if self.fabric.local_rank == 0:
        try:
            os.remove(metric_sync_path)
        except FileNotFoundError:
            pass
    self.fabric.barrier()  # ✅ 只在 metric 启动前同步

# ... extract 和 cache 逻辑 ...

if self.fabric.local_rank == 0:
    try:
        result = self.compute_metric(collection, collection_flip)
        self.log(result, epoch, step, n_images_seen)
        if metric_sync_path is not None:
            self._publish_metric_status(metric_sync_path, "ok")  # 发布成功
    except Exception as error:
        if metric_sync_path is not None:
            self._publish_metric_status(metric_sync_path, "error", error)  # 发布失败
        raise
    finally:
        del collection, collection_flip
else:
    result = {}
    if not use_cache:
        del collection, collection_flip
    if metric_sync_path is not None:
        torch.cuda.empty_cache()
        self._wait_for_metric_status(metric_sync_path)  # ✅ 轮询而非 NCCL barrier

# ✅ metric 完成后才执行 barrier
self.fabric.barrier()
torch.cuda.empty_cache()
return result
```

**特性**：
- 只对 `type='4'` 的评估器启用（大规模评估使用 `get_sim_matrix_large_scale_v5(num_gpus=8)`）
- 状态文件路径包含 `epoch/step/n_images_seen`，避免冲突
- 原子写入（先写临时文件再 `os.replace`）
- 超时机制（环境变量 `QGFACE_METRIC_SYNC_TIMEOUT_SEC`）
- 错误传播（rank 0 失败会通知其他 rank）

---

### 2. Combined Evaluations

#### qgface（旧版）- ❌ 有死锁风险

```python
# qgface/train_opt.py:643-658
combined_config = getattr(cfg.evaluations, 'combined_evaluations', None)
if combined_config:
    print(f'[Rank {fabric.local_rank}] 等待合并评估 (rank 0 计算中)...')
    if fabric.local_rank == 0:
        evaluators_dict = {e.name: e for e in evaluators}
        combined_start = time.time()
        combined_result = run_combined_evaluations(evaluators_dict, combined_config)
        # ↑ 内部调用 get_sim_matrix_large_scale_v4(num_gpus=8) → ThreadPoolExecutor
        all_result.update(combined_result)
        print(f'合并评估完成，耗时: {(time.time() - combined_start) / 60:.2f} mins')
    fabric.barrier()  # ❌ 与 CustomVerificationEvaluator 相同的死锁风险
```

#### qgface_subcenter（新版）- ✅ 已修复

```python
# qgface_subcenter/train_opt.py:643-654
combined_config = getattr(cfg.evaluations, 'combined_evaluations', None)
if combined_config:
    print(f'[Rank {fabric.local_rank}] 等待合并评估 (rank 0 计算中)...')
    evaluators_dict = {e.name: e for e in evaluators}
    combined_start = time.time()
    combined_result = run_combined_evaluations_distributed(
        fabric, evaluators_dict, combined_config, epoch, step, n_images_seen
    )  # ✅ 所有 rank 都调用，内部处理同步
    all_result.update(combined_result)
    if fabric.local_rank == 0:
        print(f'合并评估完成，耗时: {(time.time() - combined_start) / 60:.2f} mins')
# ✅ 不需要显式 barrier，run_combined_evaluations_distributed 内部已处理
```

**新增函数**（`qgface_subcenter/evaluations/__init__.py:129-141`）：

```python
def run_combined_evaluations_distributed(
    fabric, evaluators_dict, combined_config, epoch, step, n_images_seen
):
    """Run combined GPU metrics without parking non-zero ranks in NCCL."""
    if not evaluators_dict:
        return {}
    coordinator = next(iter(evaluators_dict.values()))
    return coordinator.run_rank_zero_metric(
        lambda: run_combined_evaluations(evaluators_dict, combined_config),
        epoch,
        f"combined_{step}",
        n_images_seen,
    )
```

**新增方法**（`qgface_subcenter/evaluations/custom_verification_evaluator.py:633-687`）：

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
    metric_sync_path = self._metric_sync_path(epoch, name_suffix, n_images_seen)
    
    if self.fabric.local_rank == 0:
        try:
            os.remove(metric_sync_path)
        except FileNotFoundError:
            pass
    self.fabric.barrier()
    
    if self.fabric.local_rank == 0:
        try:
            start_time = time.time()
            result = compute_fn()
            elapsed = time.time() - start_time
            print(f"Rank 0 metric '{name_suffix}' completed in {elapsed:.2f}s")
            self._publish_metric_status(metric_sync_path, "ok")
        except Exception as error:
            self._publish_metric_status(metric_sync_path, "error", error)
            raise
    else:
        result = {}
        self._wait_for_metric_status(metric_sync_path)
    
    self.fabric.barrier()
    torch.cuda.empty_cache()
    return result
```

**作用**：
- 通用的 rank-0-only 计算包装器
- 复用 `_metric_sync_path/_publish_metric_status/_wait_for_metric_status` 辅助方法
- 用于任何需要 rank 0 启动多 GPU 计算的场景

---

## 配置启用情况

使用 `combined_evaluations` 的配置文件：

```yaml
# qgface_subcenter/evaluations/configs/val_20260320.yaml
per_epoch_evaluations:
  "work_0320_3t":
    evaluation_type: 'custom_verification4'  # type='4' 触发状态文件同步
  "work_0320_glint":
    evaluation_type: 'custom_verification4'

combined_evaluations:
  "work_0320":
    sources: ["work_0320_3t", "work_0320_glint"]
    # ↑ 触发 run_combined_evaluations_distributed
```

`test_20260320.yaml` 也有类似配置。

---

## 其他修改

### GC 和缓存清理

- 两版本都保留了 `torch.cuda.empty_cache()` 和 `gc.collect()`
- 文档明确：**不要删除 GC**，首次评估的耗时差异来自文件系统缓存冷启动（55-64秒），而非 GC

### 文件数量对比

```bash
qgface/evaluations/custom_verification_evaluator.py:         916 行
qgface_subcenter/evaluations/custom_verification_evaluator.py: 1037 行
差异: +121 行（新增辅助方法 + run_rank_zero_metric + 修改 evaluate 逻辑）
```

---

## 修复状态总结

| 修复项 | qgface | qgface_subcenter | 状态 |
|--------|--------|-------------------|------|
| **CustomVerificationEvaluator barrier** | ❌ 直接 barrier | ✅ 状态文件同步（type='4'） | **✅ 已修复** |
| **Combined evaluations barrier** | ❌ 直接 barrier | ✅ `run_combined_evaluations_distributed` | **✅ 已修复** |
| **`run_rank_zero_metric` 实现** | ❌ 不存在 | ✅ 已实现（633-687行） | **✅ 已实现** |
| **辅助方法（sync/publish/wait）** | ❌ 不存在 | ✅ 已实现（595-631行） | **✅ 已实现** |
| **`run_combined_evaluations_distributed`** | ❌ 不存在 | ✅ 已实现（`__init__.py:129-141`） | **✅ 已实现** |

---

## 验证清单

使用 `qgface_subcenter` 时应验证：

- [x] `custom_verification_evaluator.py` 包含 `_metric_sync_path/_publish_metric_status/_wait_for_metric_status` 辅助方法
- [x] `custom_verification_evaluator.py` 包含 `run_rank_zero_metric` 方法
- [x] `evaluate` 方法中 `type='4'` 时使用状态文件同步
- [x] `evaluations/__init__.py` 包含 `run_combined_evaluations_distributed` 函数
- [x] `train_opt.py` 调用 `run_combined_evaluations_distributed` 而非直接调用 `run_combined_evaluations`
- [ ] 实际运行多 GPU 训练，确认没有 NCCL timeout
- [ ] 实际运行 `combined_evaluations`，确认 rank 1-7 轮询文件期间 GPU 可被 rank 0 使用
- [ ] 检查 `/tmp/qgface_metric_sync/` 目录中的状态文件是否正确生成和清理

---

## 调试建议

### 环境变量

```bash
# 延长 NCCL timeout（用于调试，默认 30 分钟）
export NCCL_TIMEOUT=7200

# 调整状态文件轮询超时（默认 7200 秒）
export QGFACE_METRIC_SYNC_TIMEOUT_SEC=10800

# 启用 NCCL 异步错误处理
export NCCL_ASYNC_ERROR_HANDLING=1
```

### 日志监控

```bash
# 监控状态文件生成
watch -n 1 'ls -lh /tmp/qgface_metric_sync/*/'

# 查看 rank 0 的 metric 日志
grep "Rank 0 metric" <log_file>

# 查看所有 rank 的等待状态
grep "等待合并评估" <log_file>
```

### 常见问题

**问题 1**: `AttributeError: 'CustomVerificationEvaluator' object has no attribute 'run_rank_zero_metric'`
- **原因**: 方法未实现或版本错误
- **解决**: 确认使用 `qgface_subcenter` 且 `custom_verification_evaluator.py` 已更新

**问题 2**: 状态文件超时 `TimeoutError: Timed out waiting for metric status`
- **原因**: rank 0 的 metric 计算时间超过 7200 秒
- **解决**: 设置 `export QGFACE_METRIC_SYNC_TIMEOUT_SEC=14400` 或检查 rank 0 是否崩溃

**问题 3**: 仍然出现 NCCL timeout
- **原因**: 可能还有其他未修复的 barrier 调用
- **解决**: 搜索 `fabric.barrier()` 并检查前后是否有多 GPU 计算

---

## 结论

**qgface_subcenter 已完成所有必要的 barrier 修复**，包括：

1. ✅ **CustomVerificationEvaluator** 的 type='4' 评估使用状态文件同步
2. ✅ **Combined evaluations** 通过 `run_combined_evaluations_distributed` 和 `run_rank_zero_metric` 避免死锁
3. ✅ **辅助方法** 完整实现了原子状态发布、轮询等待、错误传播

**建议**：
- 在生产环境使用 **qgface_subcenter**
- 保留 **qgface** 作为稳定基线参考
- 后续合并其他分支时，注意搜索 `fabric.barrier()` 并检查是否需要应用相同的修复模式

**文档参考**：
- `/root/zhaokj/CVLface/cvlface/research/recognition/code/evaluation_barrier_issue.md` - 问题描述与修复原则
- `/root/zhaokj/CVLface/run_rank_zero_metric_analysis.md` - `run_rank_zero_metric` 详细分析
- 本文档 - 完整对比与实现验证

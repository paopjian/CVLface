# 分布式评估 barrier 问题记录与修改方法

更新时间：2026-08-18

适用范围：`qgface`、`qgface_subcenter`、`work_0605`、`run_v1` 中使用 Fabric/DDP 的训练评估流程。

## 结论

当前评估的首要问题是分布式同步顺序，而不是 `gc.collect()`。

`gc.collect()` 和 `torch.cuda.empty_cache()` 暂时保留，不因为本次耗时实验删除 GC。评估顺序实验表明，首次读取大评估集时的文件系统 page cache 冷启动可以造成约 55-64 秒差异，不能把这部分差异归因于 GC。

## 故障表现

旧流程中，某些评估的 `compute_metric()` 由 rank 0 启动 8 卡 GPU worker pool，而 rank 1-7 已经进入 NCCL `barrier`：

```text
rank 1-7: 等待 barrier
rank 0:   等待自己启动的多卡 metric 使用其他 GPU
```

这会导致：

- GPU 0 进入 metric 或保持空闲；
- GPU 1-7 显存常驻并在 NCCL 中忙等；
- 进程组无法完成下一次 collective；
- 等待 NCCL timeout（可能约 30 分钟）后才报错或继续。

这不是普通的评估变慢，而是 collective 顺序不一致导致的死锁/超时。

## barrier 使用原则

### 可以保留的 barrier

以下同步点是必要的：

1. 所有 rank 完成特征提取和 CPU 文件聚合后，再开始下一阶段；
2. rank 0 的 metric 完成后，所有 rank 再继续训练；
3. 保存 checkpoint 时，确保各 rank 的文件状态一致；
4. 广播评估结果或 early-stop 状态前，确保所有 rank 处于相同控制流。

### 禁止的 barrier

不能让非零 rank 在 rank 0 执行多卡 metric 期间进入 NCCL barrier。例如：

```python
# 错误：rank 0 后续会占用其他 GPU 做 metric
if fabric.local_rank == 0:
    result = evaluator.compute_metric(collection, collection_flip)
else:
    result = {}
fabric.barrier()
```

如果 `compute_metric()` 只使用 CPU，以上模式可以工作；只要 metric 会启动多 GPU worker、调用 CUDA 或触发 NCCL，就必须改成下面的状态同步模式。

## 推荐修改模式

以唯一的 `(run, epoch, step, evaluator)` 生成状态文件路径。rank 0 执行 metric，其他 rank 不调用 NCCL barrier，只等待状态文件：

```python
sync_path = make_metric_sync_path(epoch, step, evaluator_name)

if fabric.local_rank == 0:
    remove_if_exists(sync_path)
fabric.barrier()  # 这里只用于开始前同步，metric 尚未启动

if fabric.local_rank == 0:
    try:
        result = compute_metric(collection, collection_flip)
        publish_status_atomically(sync_path, status="ok")
    except Exception as error:
        publish_status_atomically(sync_path, status="error", error=error)
        raise
else:
    wait_for_status_file(sync_path)
    result = {}

# metric 已经完全结束，所有 rank 才能重新进入 NCCL
fabric.barrier()
```

状态文件必须满足以下要求：

- 路径包含 epoch、step 和 evaluator，不能复用旧文件；
- rank 0 先写临时文件，再用 `os.replace()` 原子替换；
- 非零 rank 检查 `status == "ok"`，遇到 `"error"` 立即抛出原始异常；
- 轮询必须有超时，不能无限等待；
- metric 完成后再删除状态文件或使用下一次唯一路径。

## 当前 qgface_subcenter 状态

`evaluations/custom_verification_evaluator.py` 的 type-4 路径已经采用上述结构：

- [640-649 行](qgface_subcenter/evaluations/custom_verification_evaluator.py) 在 metric 启动前同步；
- [673-691 行](qgface_subcenter/evaluations/custom_verification_evaluator.py) rank 0 发布状态，其他 rank 轮询文件；
- [693-695 行](qgface_subcenter/evaluations/custom_verification_evaluator.py) metric 完成后才执行 barrier。

这部分逻辑不要改回“rank 0 计算、其他 rank 立即 barrier”。

## 仍需同步检查的代码

其他分支合并修改时，重点搜索以下模式：

```text
compute_metric(...)
fabric.barrier()
run_combined_evaluations(...)
torch.cuda / multiprocessing / ThreadPoolExecutor
```

特别是 `qgface_subcenter/train_opt.py` 的 `combined_evaluations`：当前 rank 0 执行 `run_combined_evaluations()`，随后所有 rank 在 [658 行](qgface_subcenter/train_opt.py) 进入 barrier。如果合并评估内部使用多 GPU，必须改为状态文件等待；如果确认完全是 CPU 计算，才可以保留当前 barrier。

## GC 与耗时实验记录

训练驻留 500 个真实 batch 后的评估结果：

```text
原顺序：baseline -> no_gc
baseline（第一个）：335.11 s
no_gc（第二个）：267.70 s

反转顺序：no_gc -> baseline
no_gc（第一个）：322.86 s
baseline（第二个）：271.06 s
```

同一 variant 放在第一个或第二个，耗时差异约 55-64 秒，且差异集中在大评估集第一次 `extract_original`。原因是首次读取触发磁盘/Arrow/LMDB 的 page cache 冷启动，第二次读取命中系统缓存。

因此：

- 保留 `gc.collect()`，不要将其作为 barrier 问题的修复手段；
- 不使用首次冷启动耗时判断 GC 是否导致减速；
- 若要单独测 GC，必须先预热所有 evaluator，再随机或交替测试 variant；
- `torch.cuda.empty_cache()` 只影响 CUDA allocator，不会恢复系统 page cache。

## 合并与验证清单

- [ ] 所有 rank 在特征提取阶段执行相同的 collective 顺序；
- [ ] rank 0 的多 GPU metric 期间，其他 rank 不进入 NCCL barrier；
- [ ] metric 成功和失败都能通知其他 rank；
- [ ] metric 结束后所有 rank 再执行同一个 barrier；
- [ ] `combined_evaluations` 已确认是 CPU-only，或已改为状态文件同步；
- [ ] 使用至少一次 warm-up 后的评估时间比较 GC；
- [ ] 测试期间确认没有 GPU 长时间 NCCL busy-wait 和 30 分钟 timeout。


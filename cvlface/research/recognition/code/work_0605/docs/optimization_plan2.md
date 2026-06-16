# train_opt.py 训练循环深度优化方案

benchmark.md 中已完成的优化: RecordIO格式、TurboJPEG解码、v2 numpy augmenter、torch.compile、channels_last、persistent_workers、prefetch_factor。

本文档聚焦**训练循环内部**剩余的性能瓶颈。

---

## P0: 高优先级 (预计合计 5-10% 训练加速)

### 1. `loss.item()` 每 batch 触发 CUDA 同步

**位置**: `train_opt.py:408`

```python
# 现状: 每个 batch 都调用 .item(), 强制 GPU→CPU 同步
epoch_losses.append(loss.item())
```

**问题**: `.item()` 等价于 `torch.cuda.synchronize()`, 阻止了 GPU forward/backward 与 CPU DataLoader prefetch 的并行重叠。这是最大的隐形瓶颈。

**修复**:
```python
# 方案: 在 GPU 上累积, 只在 logging 时 sync
epoch_loss_sum = torch.zeros(1, device=fabric.device)
epoch_loss_count = 0

# 循环内:
epoch_loss_sum += loss.detach()
epoch_loss_count += 1

# 在 200 batch logging 时:
avg_loss = (epoch_loss_sum / epoch_loss_count).item()
```

**预估收益**: 3-8% 训练速度提升 (取决于 batch 大小和 GPU 利用率)

---

### 2. `get_norm()` 逐参数 `.item()` 导致大量 GPU 同步

**位置**: `train_opt.py:46-52`

```python
# 现状: IR-101 ~200个参数, 每个参数一次 .item() = 200次 GPU sync
def get_norm(module):
    grad_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2  # sync!
    grad_norm = grad_norm ** 0.5
    return grad_norm
```

**修复**:
```python
def get_norm(module):
    norms = []
    for p in module.parameters():
        if p.grad is not None:
            norms.append(p.grad.data.norm(2))
    if not norms:
        return 0.0
    total = torch.stack(norms).norm(2)
    return total.item()  # 只一次 sync
```

或者直接用 PyTorch 内置 (已经优化过):
```python
params = [p for p in module.parameters() if p.grad is not None]
if params:
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf')).item()
```

**预估收益**: 每 200 batch 节省 ~400次 GPU sync (backbone + classifier)

---

## P1: 中优先级 (1-3% 加速, 改动极小)

### 3. `optimizer.zero_grad()` 缺少 `set_to_none=True`

**位置**: `train_opt.py:402`

```python
# 现状:
optimizer.zero_grad()
# 改为:
optimizer.zero_grad(set_to_none=True)
```

**原理**: 默认行为是对所有 grad tensor 做 memset(0)。`set_to_none=True` 直接将 `.grad` 设为 None, 跳过 memset, 下次 backward 时 autograd 直接写入新 tensor 而非累加。PyTorch 官方推荐。

**注意**: 需确认 `get_norm()` 和 `compute_update_ratio()` 能处理 `p.grad is None` 的情况 (当前代码已有此检查, 无需改动)。

**预估收益**: 1-3%

---

### 4. 删除 `torch.cuda.empty_cache()`

**位置**: `train_opt.py:550`

```python
# 现状: 每 epoch 结束
torch.cuda.empty_cache()
```

**问题**:
- 强制 CUDA sync + 归还所有未使用缓存块到 OS
- 下一 epoch 开头 PyTorch allocator 需重新分配 → 碎片化 + 冷启动延迟
- 在 eval 前调用有意义, 但 eval 结束后紧接训练时是有害的

**修复**: 删除此行。如果担心 eval 阶段 OOM, 可以只在 eval 开始前调用。

---

## P2: 改善型 (减少资源浪费, 量化收益较小)

### 5. `param_cache` 整模型克隆

**位置**: `train_opt.py:389-398`

```python
# 每 200 batch: 克隆 IR-101 全部可训练参数 (~65M params × 4B = 260MB 显存)
param_cache_backbone = {
    name: p.data.clone()
    for name, p in model.named_parameters() if p.requires_grad
}
```

`compute_update_ratio()` (line 54-67) 也逐参数 `.item()`:
```python
total_delta_norm += delta.norm(2).item() ** 2   # sync per param
total_param_norm += param_cache[name].norm(2).item() ** 2  # sync per param
```

**修复方案** (选一):
- A) 降低频率: 每 1000 batch 或每 epoch 只算一次
- B) 只追踪最后一层 norm 变化作为 proxy
- C) GPU 上累积 norm (与 get_norm 同样的手法)
- D) 如果不需要此诊断指标, 直接删除

**推荐**: 方案 A + C 组合

---

### 6. `compute_update_ratio()` 同样有逐参数 sync 问题

**位置**: `train_opt.py:54-67`

修复方式与 `get_norm()` 相同: GPU 上累积再一次 `.item()`。

---

### 7. tqdm 每 batch 更新

**位置**: `train_opt.py:448-449`

```python
pbar.set_description(f"Epoch {epoch} | ...")
pbar.update(1)
```

**问题**: tqdm 每次 update 调用 `time.time()` + 字符串格式化 + `sys.stdout.write`。37M 图/epoch ÷ 512/batch ÷ 7GPU = ~10,300 batch/epoch, 每个都做 stdout I/O。

**修复**:
```python
if batch_idx % 10 == 0:
    pbar.set_description(...)
    pbar.update(min(10, batch_length - pbar.n))
```

---

### 8. `get_last_lr()` 每 batch 用 numpy 算均值

**位置**: `optims/lr_scheduler.py:82-84`

```python
def get_last_lr(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    return float(np.mean(lrs))  # numpy for 2 numbers...
```

两个 param_group 的 lr 是同一个 scheduler 设置的, 永远相等。

**修复**:
```python
def get_last_lr(optimizer):
    return optimizer.param_groups[0]['lr']
```

---

## P3: 低优先级 (架构层面, 需要更多改动)

### 9. MLflow SQLite 同步写入阻塞训练线程

**位置**: `train_opt.py:437-444`

每 200 batch 在主线程做 SQLite INSERT, 有磁盘 I/O 阻塞风险。

**修复**: 异步写入
```python
import threading, queue

_mlflow_queue = queue.Queue()

def _mlflow_writer():
    while True:
        item = _mlflow_queue.get()
        if item is None:
            break
        metrics, step = item
        mlflow.log_metrics(metrics, step=step)

_mlflow_thread = threading.Thread(target=_mlflow_writer, daemon=True)
_mlflow_thread.start()

# 使用时:
_mlflow_queue.put((mlflow_metrics, step))
```

---

### 10. `augment_dataset_v2.py` 中的冗余 `.copy()`

**位置**: `augment_dataset_v2.py:47`

```python
sample_t = torch.from_numpy(aug_np.transpose(2, 0, 1).copy()).float()
```

`transpose` 返回非连续视图 → `.copy()` 做一次 uint8 拷贝 → `.float()` 再做一次 float32 分配。

**修复**:
```python
# 跳过 uint8 拷贝, 直接在 float 阶段处理连续性
sample_t = torch.from_numpy(np.ascontiguousarray(aug_np.transpose(2, 0, 1))).float()
```

或者 (避免 numpy 拷贝, 让 torch 处理):
```python
sample_t = torch.as_tensor(aug_np).permute(2, 0, 1).contiguous().float()
```

对 112×112 小图实际差异很小, 但语义更清晰。

---

### 11. `speed` 计算包含 logging 开销

**位置**: `train_opt.py:446-450`

```python
speed = cfg.trainers.batch_size / (time.time() - tic)
speed_total = speed * fabric.world_size
pbar.set_description(...)
pbar.update(1)
tic = time.time()  # tic 在循环末尾更新, 包含了 set_description 开销
```

**修复**: 把 `tic = time.time()` 移到循环开头 (forward 之前), 或用滑动平均。

---

## 实施顺序建议

```
Phase 1 (改动最小, 收益最大):
  - #1 loss.item() → 延迟 sync
  - #3 zero_grad(set_to_none=True)
  - #4 删除 empty_cache()

Phase 2 (改 get_norm / compute_update_ratio):
  - #2 get_norm GPU 累积
  - #5/#6 param_cache 降频 + GPU 累积

Phase 3 (polish):
  - #7 tqdm 降频
  - #8 get_last_lr 简化
  - #10 .copy() 优化
  - #9 MLflow 异步 (如果 profiling 显示有 I/O 阻塞)
```

---

## 已排除的优化

以下方案经分析后**不推荐**:

| 方案 | 原因 |
|------|------|
| pin_memory_device='cuda:X' | PyTorch 2.x DataLoader 已自动优化, 手动指定无额外收益 |
| CUDA Graph | 动态 shape (partial_fc 采样) 不兼容 |
| Gradient checkpointing | 24GB 显存对 IR-101 够用, 不值得用计算换显存 |
| 增大 batch_size | 已经 512/GPU, 受限于 partial_fc 显存 |
| 更多 num_workers | benchmark 结果显示 w=8 和 w=16 差异不大, CPU 已饱和 |

---

## 验证方法

改动后用 `trainers.limit_num_batch=1000` 跑短训练对比:

```bash
# baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run --strategy=ddp --devices=7 --precision="bf16-mixed" \
    train_opt.py trainers.num_gpu=7 trainers.limit_num_batch=1000 ...

# 优化后
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run --strategy=ddp --devices=7 --precision="bf16-mixed" \
    train_opt2.py trainers.num_gpu=7 trainers.limit_num_batch=1000 ...
```

对比 Epoch Time 和 Avg Speed。

---

## PartialFC 分类器优化 (classifiers/partial_fc/partial_fc.py)

### P0: `DistCrossEntropyFunc.backward` 里的 `.item()` — 每次 backward 都 sync GPU

**位置**: `partial_fc.py:248`

```python
# 之前: 每个 batch backward 都触发 GPU→CPU sync
return logits * loss_gradient.item(), None

# 之后: 0-dim GPU tensor 直接广播乘, 零 sync
return logits * loss_gradient, None
```

**预估收益**: 3-5% 全训练加速 (每 batch backward 必经路径)

### P1: `sample()` 冗余 `.cuda()` 调用

**位置**: `partial_fc.py:109-114`

```python
# 之前: 4 个冗余 .cuda() (tensor 已在 GPU, 纯 dispatch overhead)
positive = torch.unique(labels[index_positive], sorted=True).cuda()
perm = torch.rand(size=[self.num_local]).cuda()
index = torch.topk(perm, k=self.num_sample)[1].cuda()
index = index.sort()[0].cuda()

# 之后: 去掉冗余调用, torch.rand 直接指定 device
positive = torch.unique(labels[index_positive], sorted=True)
perm = torch.rand(size=[self.num_local], device=labels.device)
index = torch.topk(perm, k=self.num_sample)[1]
index = index.sort()[0]
```

### P2: `_gather_labels` 使用 `.long().cuda()` → 直接 dtype+device 参数

```python
# 之前:
torch.zeros(batch_size).long().cuda()

# 之后:
torch.zeros(batch_size, dtype=torch.long, device=local_labels.device)
```

### P3: 去掉 `logits.clamp(-1, 1)` 和 embedding 手动除法

- `logits.clamp(-1, 1)` 是多余防御 (normalized dot product 数学保证 [-1,1])
- `embeddings / norms` 改为 `normalize(embeddings, dim=1)` (PyTorch 内部 fused kernel)
- `norms` 保留给 AdaFace 用

---

## 已完成改动汇总

| 文件 | 改动 | 类型 |
|------|------|------|
| `train_opt.py` | `get_norm()` → fused `clip_grad_norm_` | GPU sync 优化 |
| `train_opt.py` | `compute_update_ratio()` → GPU 上累积 | GPU sync 优化 |
| `train_opt.py` | `loss.item()` → GPU fp32 累积 | GPU sync 优化 |
| `train_opt.py` | pbar `loss` 格式化降频到每 10 batch | GPU sync 优化 |
| `train_opt.py` | `zero_grad(set_to_none=True)` | memset 优化 |
| `train_opt.py` | MLflow 异步后台线程写入 | I/O 阻塞优化 |
| `partial_fc.py` | `loss_gradient.item()` → 直接 tensor 乘 | backward sync 优化 |
| `partial_fc.py` | 去掉冗余 `.cuda()` | dispatch overhead |
| `partial_fc.py` | `_gather_labels` device 参数 | 代码规范 |
| `partial_fc.py` | 去掉 `logits.clamp(-1,1)` | 多余 kernel |
| `partial_fc.py` | embedding normalize 用 `F.normalize` | fused kernel |

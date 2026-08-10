# 8卡4090训练加速优化方案

## 当前配置
- 硬件: 8x RTX 4090 (24GB, PCIe, 无NVLink)
- 框架: Lightning Fabric + DDP
- 精度: bf16-mixed
- 模型: IR101 (iresnet101)
- batch_size: 256 per GPU, total 2048
- num_workers: 8
- 优化器: SGD, lr=0.004
- 分类器: partial_fc, sample_rate=0.40
- 梯度裁剪: max_grad_norm=1.0

---

## 优化项列表

### 1. 数据加载优化
- **状态**: [ ] 待讨论
- **预期加速**: 显著（尤其IO瓶颈时）
- **优先级**: 高
- **内容**:
  - num_workers 从 8 提升到 16+（当前8卡各分1个worker，不够）
  - persistent_workers=True（避免每epoch重新fork）
  - prefetch_factor=3~4（提前预取更多batch）
  - pin_memory=True（加速CPU→GPU传输）
- **风险**: 内存占用增加，需确认CPU内存充足

### 2. torch.compile 编译加速
- **状态**: [ ] 待讨论
- **预期加速**: 10-30%
- **优先级**: 高
- **内容**:
  - 对model使用 `torch.compile(model, mode="reduce-overhead")`
  - 4090 Ada架构对torch.compile支持好
  - 人脸识别输入尺寸固定，适合CUDA Graphs
- **风险**: 首次编译耗时；部分动态shape操作可能不兼容
- **位置**: train3.py:280 附近，model setup之前

### 3. cudnn.benchmark
- **状态**: [ ] 待讨论
- **预期加速**: 5-15%
- **优先级**: 高
- **内容**:
  - 添加 `torch.backends.cudnn.benchmark = True`
  - 输入尺寸固定时cuDNN自动选择最快卷积算法
- **风险**: 几乎无风险
- **位置**: train3.py 脚本开头

### 4. 增大 per-GPU batch_size
- **状态**: [ ] 待讨论
- **预期加速**: 提升GPU计算利用率
- **优先级**: 中
- **内容**:
  - 当前 batch_size=256，bf16下4090(24GB)可能有余量
  - 可试探增加到 320 或 384
  - 学习率需线性缩放: lr_new = lr_old × (new_batch / old_batch)
- **风险**: 显存不足导致OOM；大batch可能影响收敛

### 5. 梯度累积减少DDP通信
- **状态**: [ ] 待讨论
- **预期加速**: 减少allreduce通信开销
- **优先级**: 中
- **内容**:
  - 代码已支持 gradient_acc（train3.py:374）
  - 设置 trainers.gradient_acc=2，每2步同步一次
  - 等效batch翻倍，需调整lr
- **风险**: 等效batch过大可能影响收敛质量

### 6. 减少不必要的同步和拷贝
- **状态**: [ ] 待讨论
- **预期加速**: 减少不必要开销
- **优先级**: 中
- **内容**:
  - train3.py:386-395 每50步clone全部参数计算update_ratio
  - IR101参数量大，clone开销不可忽略
  - 可降低频率到200步，或直接移除（update_ratio对训练无实际影响）
- **风险**: 无风险，纯监控指标

### 7. NCCL 通信调参
- **状态**: [ ] 待讨论
- **预期加速**: PCIe场景有一定帮助
- **优先级**: 低
- **内容**:
  - 8卡4090走PCIe，DDP梯度同步是瓶颈之一
  - 设置环境变量优化NCCL行为:
    ```
    NCCL_P2P_DISABLE=0
    NCCL_IB_DISABLE=1
    NCCL_SOCKET_IFNAME=lo
    NCCL_ALGO=Ring
    ```
- **风险**: 需要根据实际拓扑调试

### 8. 降低评估频率
- **状态**: [ ] 待讨论
- **预期加速**: 减少总训练时间中评估占比
- **优先级**: 低
- **内容**:
  - 如果当前每epoch都评估，可改为每3-5个epoch
  - 通过 evaluations.eval_every_n_epochs 控制
- **风险**: 可能错过最佳checkpoint

---

## 实施记录

| 序号 | 优化项 | 状态 | 实施日期 | 效果 |
|------|--------|------|----------|------|
| 1 | 数据加载优化 | 已实施 | 2026-02-24 | persistent_workers+prefetch_factor=3, num_workers保持8 |
| 2 | torch.compile | 已实施 | 2026-02-24 | mode=default, 仅编译model不编译classifier |
| 3 | cudnn.benchmark | 已实施 | 2026-02-24 | torch.backends.cudnn.benchmark=True |
| 4 | 增大batch_size | 跳过 | 2026-02-24 | 显存余量不足，风险大 |
| 5 | 梯度累积 | 跳过 | 2026-02-24 | batch已2048，再翻倍收敛风险高 |
| 6 | 减少同步/拷贝 | 已实施 | 2026-02-24 | param_cache clone和update_ratio频率从50步降为200步 |
| 7 | NCCL调参 | 跳过 | 2026-02-24 | 单机场景默认配置已足够，收益极小 |
| 8 | 降低评估频率 | 跳过 | 2026-02-24 | 用户需要每epoch评估 |

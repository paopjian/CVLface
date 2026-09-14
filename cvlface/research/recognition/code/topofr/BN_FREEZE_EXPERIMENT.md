# BN 冻结对比实验

## 实验概述

验证 Batch Normalization (BN) 层冻结对预训练模型微调的影响。

### 问题背景

你观察到一个关键现象：
- **预训练模型直接评估**: agedb_30 = 98.63% ✓
- **训练 Epoch 0 后**: agedb_30 ≈ 50% (接近随机)
- **训练 Epoch 1 后**: agedb_30 = 72.07%

这表明预训练模型的特征提取能力在微调早期就被破坏了。

### 假设

BN 层的 running_mean 和 running_var 统计量在训练过程中被新数据集的统计量覆盖，导致：
1. 特征分布发生剧烈变化
2. 预训练的特征提取能力部分失效
3. 需要重新训练才能恢复性能

### 实验设计

**对比两组实验**:

1. **冻结 BN 组**: 
   - 冻结所有 BN 层的 running_mean/running_var (设为 eval 模式)
   - 冻结 BN 的 gamma/beta 参数
   - 预期：特征分布保持稳定，性能维持在 ~98%

2. **不冻结 BN 组** (标准微调):
   - BN 层正常训练，统计量正常更新
   - 预期：特征分布变化，性能在早期下降到 ~72%

**评估节点**: 训练进度 0%, 10%, 20%, 30%, 40%, 50% 时评估 agedb_30

## 文件说明

### 1. `train_bn_freeze_experiment.py`
核心实验脚本，特点：
- 接受 `freeze_bn=True/False` 参数
- 自动加载预训练模型 (Glint360K_R100_TopoFR)
- 在 0%/10%/20%/30%/40%/50% 进度评估 agedb_30
- 保存结果到 JSON 文件

### 2. `scripts/train/run_bn_freeze_experiment.sh`
自动化运行脚本：
- 顺序运行冻结 BN 和不冻结 BN 两组实验
- 汇总结果并生成对比报告
- 8 GPU 并行训练

### 3. `evaluations/configs/agedb30_only.yaml`
轻量级评估配置，只评估 agedb_30 以加快实验速度

### 4. `monitor_bn_experiment.sh`
实验监控脚本，显示进度和结果

## 运行方式

### 自动化运行（推荐）
```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr
bash scripts/train/run_bn_freeze_experiment.sh
```

### 手动运行单个实验
```bash
# 冻结 BN
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
    --devices=8 --precision="bf16-mixed" \
    train_bn_freeze_experiment.py \
    freeze_bn=True

# 不冻结 BN
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run \
    --devices=8 --precision="bf16-mixed" \
    train_bn_freeze_experiment.py \
    freeze_bn=False
```

### 监控进度
```bash
# 实时监控
bash monitor_bn_experiment.sh

# 或持续监控（每 10 秒刷新）
watch -n 10 bash monitor_bn_experiment.sh

# 或查看原始日志
tail -f bn_experiment.log
```

## 实验配置

- **预训练模型**: Glint360K_R100_TopoFR (IJB-C 97.60%)
- **微调数据集**: dataset_0605 (791,509 类)
- **架构**: IResNet-101
- **训练配置**: 
  - 8x GPU, batch_size=128/GPU
  - bf16 mixed precision
  - lr=0.0005, SGD with momentum
  - 15 epochs total (实验只训练到 50% 进度)
- **评估**: agedb_30 verification

## 预期结果

### 冻结 BN 组
```
0%:  98.63%  (预训练基准)
10%: ~98%    (保持稳定)
20%: ~98%
30%: ~98%
40%: ~98%
50%: ~98%
```

### 不冻结 BN 组
```
0%:  98.63%  (预训练基准)
10%: ~72%    (性能下降！)
20%: ~75%    (开始恢复)
30%: ~80%
40%: ~85%
50%: ~90%
```

## 输出文件

结果保存在 `/data1/dataset_0605/train_output/bn_freeze_exp_*/`:

- `bn_freeze_results_frozen.json` - 冻结 BN 实验结果
- `bn_freeze_results_unfrozen.json` - 不冻结 BN 实验结果

JSON 格式：
```json
{
  "freeze_bn": true/false,
  "results": {
    "0%": 0.9863,
    "10%": 0.9801,
    ...
  },
  "full_results": {...}
}
```

## 实验状态

- **创建时间**: 2026-09-02
- **当前状态**: 运行中（后台进程）
- **预计时长**: 每组实验约 30-60 分钟（取决于数据集大小）

## 后续行动

根据实验结果：

1. **如果冻结 BN 有效** (性能保持稳定):
   - 建议微调时默认冻结 BN
   - 或采用渐进式解冻策略
   - 更新训练脚本添加 BN 冻结选项

2. **如果不冻结 BN 最终性能更好**:
   - 接受早期性能下降是必要的
   - 使用更小学习率 + 更长 warmup
   - 延长训练周期让模型充分恢复

3. **对比最终性能** (50% 训练后):
   - 如果两组性能接近 → 冻结 BN 更高效（避免早期崩溃）
   - 如果不冻结 BN 显著更好 → 需要完整训练周期

## 技术细节

### BN 冻结实现
```python
def freeze_bn_layers(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            module.eval()  # 固定 running_mean/var
            for param in module.parameters():
                param.requires_grad = False  # 冻结 gamma/beta
```

### 训练进度控制
训练到指定百分比而非固定 epoch，确保两组实验训练量完全相同。

### DDP 同步
所有评估前都有 `fabric.barrier()` 确保多 GPU 同步。

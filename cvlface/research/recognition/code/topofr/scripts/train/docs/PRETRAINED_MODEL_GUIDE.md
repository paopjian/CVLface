# 使用 TopoFR 预训练模型训练指南

## 📦 预训练模型信息

**模型路径**: `/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt`

**模型详情**:
- **训练数据**: Glint360K (~360K 类别, ~17M 图片)
- **架构**: IResNet-100
- **损失函数**: TopoFR (ArcFace + 拓扑对齐 + GUM加权)
- **性能**: 
  - IJB-C @ FAR=1e-5: **96.57%**
  - IJB-C @ FAR=1e-4: **97.60%**

这是一个**已经用 TopoFR 训练好的高性能模型**，在 Glint360K 大规模数据集上训练完成。

---

## 🎯 使用场景

### 场景1: Fine-tuning 到 dataset_0605 ⭐ 推荐

从这个高质量的 TopoFR 预训练模型开始，在你的数据集上 fine-tune：

**优势**:
- ✅ 起点更高（96.57% vs 从头开始）
- ✅ 收敛更快
- ✅ 最终性能更好
- ✅ 训练时间更短

### 场景2: 直接使用进行推理

如果只需要评估性能，可以直接使用预训练模型。

---

## 🚀 三个训练脚本

### 1. train_topofr_from_pretrained.sh ⭐ 推荐

**标准 fine-tuning 脚本**

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr
bash train_topofr_from_pretrained.sh
```

**配置**:
- **Epochs**: 15
- **学习率**: 0.0005 (较低，适合 fine-tuning)
- **LR milestones**: [8, 12]
- **时间**: ~20 小时
- **用途**: 标准 fine-tuning，保守策略

### 2. train_topofr_from_pretrained_quick.sh

**快速验证版本**

```bash
bash train_topofr_from_pretrained_quick.sh
```

**配置**:
- **Epochs**: 5
- **学习率**: 0.0005
- **时间**: ~6 小时
- **用途**: 快速验证代码和配置

### 3. train_topofr_from_pretrained_aggressive.sh

**激进学习率版本**

```bash
bash train_topofr_from_pretrained_aggressive.sh
```

**配置**:
- **Epochs**: 20
- **学习率**: 0.001 (较高)
- **LR milestones**: [10, 15, 18]
- **时间**: ~25 小时
- **用途**: 数据分布差异大时使用

---

## 📊 训练策略对比

| 策略 | 学习率 | Epochs | 适用场景 |
|------|--------|--------|---------|
| **标准 fine-tune** | 0.0005 | 15 | 数据分布相似，保守训练 ⭐ |
| **快速验证** | 0.0005 | 5 | 代码验证、快速实验 |
| **激进 fine-tune** | 0.001 | 20 | 数据分布差异大 |
| **从头训练** | 0.001 | 20+ | 无预训练或完全不同领域 |

---

## 🔧 参数说明

### 核心差异

与从头训练相比，使用预训练模型时的关键差异：

#### 1. 模型架构
```bash
# 注意：必须使用 IResNet-100，与预训练模型匹配
models=iresnet/configs/v1_ir100.yaml  # R100, 不是 R101!

# 预训练模型路径
models.start_from=/root/.../Glint360K_R100_TopoFR_9760.pt
```

#### 2. 学习率策略
```bash
# Fine-tuning 使用较低学习率
optims.lr=0.0005  # 而不是 0.001

# 较少的 warmup
optims.warmup_epoch=2

# 调整 milestones
optims.lr_milestones=[8,12]  # 对应较少的 epochs
```

#### 3. 训练周期
```bash
# Fine-tuning 需要更少的 epochs
optims.num_epoch=15  # 而不是 20-30
```

---

## 💡 最佳实践

### 推荐流程

```bash
# Step 1: 快速验证 (5 epochs, ~6h)
bash train_topofr_from_pretrained_quick.sh

# 检查验证集性能，如果效果好：

# Step 2: 标准 fine-tuning (15 epochs, ~20h)
bash train_topofr_from_pretrained.sh
```

### 学习率选择指南

#### 使用 0.0005 (保守) 当：
- ✅ 预训练数据集与目标数据集相似
- ✅ 希望保留预训练模型的大部分知识
- ✅ 目标数据集较小

#### 使用 0.001 (激进) 当：
- ✅ 预训练数据集与目标数据集差异大
- ✅ 目标数据集较大 (如 dataset_0605 的 37M 图片)
- ✅ 希望模型更快适应新数据分布

---

## 📈 预期效果

### Baseline (从头训练)
- AdaFace + 0 → **94.5%** (20 epochs)
- TopoFR + 0 → **95.0%** (20 epochs)

### Fine-tuning (TopoFR预训练)
- TopoFR + fine-tune (保守) → **97.0%+** (15 epochs) 🎯
- TopoFR + fine-tune (激进) → **97.5%+** (20 epochs) 🚀

**预期提升**: +2-3% (相比从头训练)

---

## 🔍 监控要点

### 训练初期 (Epoch 0-2)

**正常现象**:
- Loss 快速下降
- 验证集准确率从高起点开始（因为预训练）
- 学习率较低但收敛快

**异常信号**:
- Loss 上升 → 学习率可能过高
- 准确率下降 → 可能破坏了预训练知识

### 训练中期 (Epoch 3-10)

**正常现象**:
- Loss 平稳下降
- 验证集准确率稳步提升
- 梯度范数稳定

### 训练后期 (Epoch 10+)

**正常现象**:
- Loss 收敛
- 验证集准确率达到峰值
- 可能需要降低学习率

---

## ⚙️ 自定义调整

### 调整学习率

编辑脚本中的：
```bash
optims.lr=0.0005  # 改为 0.0003 (更保守) 或 0.001 (更激进)
```

### 调整 epochs

```bash
optims.num_epoch=15  # 改为 10 (更快) 或 20 (更充分)
```

### 调整 batch size

如果遇到 OOM：
```bash
trainers.batch_size=128  # 从 256 减小
```

---

## 🆚 对比实验建议

### 实验组设计

1. **Baseline**: 从头训练 (AdaFace)
   ```bash
   bash train_baseline_adaface_e2e.sh
   ```

2. **TopoFR 从头**: 从头训练 (TopoFR)
   ```bash
   bash train_topofr_e2e_full.sh
   ```

3. **TopoFR Fine-tune**: 预训练 fine-tune ⭐
   ```bash
   bash train_topofr_from_pretrained.sh
   ```

### 预期结果

| 方法 | 起点 | 训练时间 | 最终性能 | 相对提升 |
|------|------|---------|---------|---------|
| AdaFace 从头 | 随机 | ~18h | 94.5% | baseline |
| TopoFR 从头 | 随机 | ~25h | 95.0% | +0.5% |
| TopoFR Fine-tune | 96.57% | ~20h | 97.0%+ | +2.5% 🎯 |

---

## ⚠️ 注意事项

### 1. 模型架构匹配

**必须使用 IResNet-100**，不能用 IResNet-101：

```bash
# 正确 ✓
models=iresnet/configs/v1_ir100.yaml

# 错误 ✗
models=iresnet/configs/v1_ir101.yaml
```

### 2. 损失函数一致性

预训练模型是用 TopoFR 训练的，建议继续使用：

```bash
# 推荐 ✓
losses=configs/topofr_adaface.yaml

# 可以用但可能不是最优 △
losses=configs/adaface.yaml
```

### 3. 学习率不要过高

Fine-tuning 时学习率过高会破坏预训练知识：

```bash
# 安全范围 ✓
optims.lr=0.0003 ~ 0.001

# 过高 ✗
optims.lr=0.005
```

---

## 🐛 故障排查

### Q1: 加载模型时报错

**错误**: `size mismatch for xxx`

**原因**: 模型架构不匹配

**解决**: 确认使用 `v1_ir100.yaml` 而非 `v1_ir101.yaml`

### Q2: Loss 突然上升

**原因**: 学习率过高

**解决**: 降低学习率到 0.0003

### Q3: 性能不如预训练模型

**原因**: Fine-tuning 破坏了预训练知识

**解决**: 
- 使用更低学习率 (0.0003)
- 减少训练 epochs
- 增加 warmup epochs

---

## 📞 快速开始

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

# 推荐：标准 fine-tuning
bash train_topofr_from_pretrained.sh
```

---

## 🎯 总结

使用 TopoFR 预训练模型的优势：

- ✅ **更高起点**: 96.57% → 97%+
- ✅ **更快收敛**: 15 epochs vs 20+ epochs
- ✅ **更好性能**: 预期提升 +2-3%
- ✅ **更稳定**: 基于大规模数据集训练

**推荐使用场景**: 几乎所有情况下都建议使用预训练模型！

祝训练顺利！🚀

# TopoFR 端到端训练脚本说明

## 📋 可用脚本列表

### 1. train_topofr_e2e_full.sh ⭐ 推荐
**完整的 TopoFR 端到端训练**
- **Epochs**: 20
- **学习率调度**: Step (milestones: [10,15,18])
- **数据增强**: gridsample_v2_numpy
- **用途**: 生产级完整训练
- **预计时间**: 根据数据集大小，约 20-30 小时

### 2. train_topofr_e2e_quick.sh
**快速验证版本**
- **Epochs**: 10
- **学习率调度**: Step (milestones: [5,8])
- **数据增强**: basic_v2_numpy (更快)
- **用途**: 代码验证、配置测试
- **预计时间**: 约 10-15 小时

### 3. train_topofr_e2e_cosine.sh
**使用 Cosine 学习率调度**
- **Epochs**: 30
- **学习率调度**: Cosine Annealing
- **数据增强**: gridsample_v2_numpy
- **用途**: 长时间训练、更平滑的收敛
- **预计时间**: 约 30-40 小时

### 4. train_baseline_adaface_e2e.sh
**Baseline 对比实验**
- **损失函数**: AdaFace (不含 TopoFR)
- **其他配置**: 与 train_topofr_e2e_full.sh 完全相同
- **用途**: 对比 TopoFR 的性能提升
- **预计时间**: 约 15-20 小时 (比 TopoFR 快 15-20%)

## 🎯 使用场景

### 场景1: 首次验证 TopoFR 集成
```bash
# 使用快速版本验证代码正确性
bash train_topofr_e2e_quick.sh
```

### 场景2: 完整训练并评估
```bash
# 先训练 baseline
bash train_baseline_adaface_e2e.sh

# 再训练 TopoFR
bash train_topofr_e2e_full.sh

# 对比两者在验证集上的性能
```

### 场景3: 追求最佳性能
```bash
# 使用 Cosine 调度器进行更长时间训练
bash train_topofr_e2e_cosine.sh
```

## 📊 配置对比表

| 参数 | Full | Quick | Cosine | Baseline |
|------|------|-------|--------|----------|
| **损失函数** | TopoFR+AdaFace | TopoFR+AdaFace | TopoFR+AdaFace | AdaFace |
| **Epochs** | 20 | 10 | 30 | 20 |
| **学习率调度** | Step | Step | Cosine | Step |
| **初始学习率** | 0.001 | 0.001 | 0.001 | 0.001 |
| **Warmup** | 2 epochs | 1 epoch | 3 epochs | 2 epochs |
| **数据增强** | gridsample | basic | gridsample | gridsample |
| **Batch Size** | 256 | 256 | 256 | 256 |
| **训练时间** | ~25h | ~12h | ~35h | ~18h |

## ⚙️ 核心参数说明

### 通用参数
```bash
trainers.num_gpu=8              # GPU 数量
trainers.batch_size=256         # 每GPU的batch size (总=256×8=2048)
trainers.num_workers=8          # DataLoader workers
trainers.precision=bf16-mixed   # 混合精度训练
```

### 模型参数
```bash
models=iresnet/configs/v1_ir101.yaml  # IResNet-101 架构
models.start_from=.../model.pt        # 预训练模型路径
models.freeze=False                   # 全模型训练
```

### 损失函数参数
```bash
# TopoFR + AdaFace
losses=configs/topofr_adaface.yaml
  - base_loss: 'adaface'
  - m: 0.4                      # AdaFace margin
  - h: 0.333                    # AdaFace h
  - t_alpha: 0.01               # AdaFace momentum
  - topo_weight: 0.1            # 拓扑损失权重
  - use_gum: true               # 使用 GUM 加权

# Baseline AdaFace
losses=configs/adaface.yaml
  - m: 0.4
  - h: 0.333
  - t_alpha: 0.01
```

### 优化器参数
```bash
optims=configs/step_sgd.yaml
optims.lr=0.001                       # 初始学习率
optims.momentum=0.9                   # SGD momentum
optims.weight_decay=0.0005            # L2 正则化
optims.max_grad_norm=5.0              # 梯度裁剪
```

### 学习率调度

#### Step 调度器
```bash
optims.scheduler='step'
optims.lr_milestones=[10,15,18]       # 在这些 epoch 降低学习率
optims.lr_lambda=0.1                  # 每次乘以 0.1
```

**学习率变化**:
- Epoch 0-9: 0.001
- Epoch 10-14: 0.0001
- Epoch 15-17: 0.00001
- Epoch 18-20: 0.000001

#### Cosine 调度器
```bash
optims.scheduler='cosine'
# 学习率从 0.001 平滑降到 0
# 遵循 cosine 曲线
```

### 分类器参数
```bash
classifiers=configs/partial_fc_sample10.yaml
classifiers.sample_rate=0.40          # PartialFC 采样率
```

## 🔧 调整建议

### 如果遇到 OOM (内存不足)

#### 方案1: 减小 batch size
```bash
trainers.batch_size=128  # 或 192
```

#### 方案2: 临时关闭拓扑损失
在配置文件中设置:
```yaml
# losses/configs/topofr_adaface.yaml
topo_weight: 0  # 临时关闭
```

#### 方案3: 使用梯度累积
```bash
trainers.gradient_accumulation_steps=2
trainers.batch_size=128
# 有效 batch size = 128 × 2 = 256
```

### 如果训练速度慢

#### 优化1: 减少 workers
```bash
trainers.num_workers=4  # 减少 CPU 开销
```

#### 优化2: 使用更简单的数据增强
```bash
data_augs=configs/basic_v2_numpy.yaml  # 替代 gridsample
```

#### 优化3: 关闭 GUM 加权
```yaml
# losses/configs/topofr_adaface.yaml
use_gum: false  # 节省约 5% 时间
```

### 如果需要更快收敛

#### 策略1: 提高学习率
```bash
optims.lr=0.002  # 从 0.001 增加到 0.002
optims.warmup_epoch=3  # 增加 warmup
```

#### 策略2: 使用 AdamW
```bash
optims=configs/adamw.yaml
optims.lr=0.0001
optims.weight_decay=0.01
```

## 📈 实验建议

### 最小可行实验
1. **验证代码** (1-2 hours)
   ```bash
   # 修改 quick 脚本，只训练 1 epoch
   optims.num_epoch=1
   trainers.limit_num_batch=100  # 只跑 100 个 batch
   ```

2. **快速实验** (10-15 hours)
   ```bash
   bash train_topofr_e2e_quick.sh
   ```

3. **完整训练** (20-30 hours)
   ```bash
   bash train_topofr_e2e_full.sh
   ```

### 对比实验
```bash
# 并行运行 baseline 和 TopoFR
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_baseline_adaface_e2e.sh &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash train_topofr_e2e_full.sh &
```

## 📊 监控训练

### TensorBoard
```bash
# 在训练过程中查看
tensorboard --logdir=/data1/dataset_0605/train_output
```

### 实时日志
```bash
# 查看最新实验的日志
tail -f /data1/dataset_0605/train_output/topofr_e2e_*/logs/*.log
```

### MLflow
```bash
# 查看 MLflow 记录
cd /data1/dataset_0605/train_output/topofr_e2e_*
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 🎯 预期结果

根据 TopoFR 论文，在相似规模数据集上：

| 方法 | IJB-C (1e-5) | IJB-C (1e-4) | 相对提升 |
|------|-------------|-------------|---------|
| AdaFace (baseline) | 94.5% | 96.2% | - |
| TopoFR + AdaFace | 95.0% | 96.7% | +0.5% |

**注意**: 实际结果取决于数据集质量和训练配置。

## ✅ 检查清单

训练前确认:
- [ ] 数据集路径正确 (`/data1/dataset_0605/train_rec`)
- [ ] 预训练模型存在
- [ ] GPU 可用 (8 卡)
- [ ] 磁盘空间充足 (至少 100GB 用于 checkpoints)
- [ ] conda 环境激活 (`cvlface`)

训练中监控:
- [ ] Loss 正常下降
- [ ] 验证集准确率提升
- [ ] GPU 利用率 > 90%
- [ ] 没有 OOM 错误

训练后评估:
- [ ] 保存最佳模型
- [ ] 在测试集上评估
- [ ] 与 baseline 对比
- [ ] 记录实验结果

## 🚀 开始训练

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

# 选择一个脚本运行
bash train_topofr_e2e_full.sh
```

祝训练顺利！🎯

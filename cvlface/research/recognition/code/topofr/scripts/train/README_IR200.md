# IR-200 TopoFR 四阶段训练配置

## 概述

本配置支持使用预训练的 IR-200 模型进行 TopoFR 四阶段训练。IR-200 是比 IR-100 更深的网络架构。

## 模型架构对比

| 模型 | Layer1 | Layer2 | Layer3 | Layer4 | 总 Blocks | 参数量 |
|------|--------|--------|--------|--------|-----------|---------|
| IR-100 | 3 | 13 | 30 | 3 | 49 | ~65M |
| IR-200 | 6 | 26 | 60 | 6 | 98 | ~125M |

## 关键适配修改

### 1. 模型配置文件
创建了 `models/iresnet_insightface/configs/v1_ir200.yaml`：
```yaml
input_size: [3, 112, 112]
color_space: 'RGB'
name: 'ir200'
output_dim: 512
start_from: ''
freeze: False
```

### 2. 模型加载器更新
修改了 `models/iresnet_insightface/__init__.py`，添加对 IR-200 的支持：
- 导入 `iresnet200` 函数
- 在 `from_config` 方法中添加 `ir200` 分支

### 3. PEFT 系统扩展
扩展了 `pefts/__init__.py` 以支持 IR-200 的 98 个 blocks：
- 原始代码只支持 IR-100 的 49 个 blocks（body.0-48）
- 新增 IR-200 的 98 个 blocks 映射（body.0-97）
- 添加直接 layer 访问支持（layer1.X, layer2.X, layer3.X, layer4.X）

### 4. 阶段 2 解冻点调整
由于 IR-200 有 98 个 blocks（IR-100 只有 49 个），阶段 2 的部分解冻点从 `body.36` 调整为 `body.72`：

| 模型 | 解冻点 | 计算 | 比例 | 实际解冻层 |
|------|--------|------|------|------------|
| **IR-100** | body.36 | 3+13+20 | 36/49=73.5% | layer3.20-29 + layer4 |
| **IR-200** | body.72 | 6+26+40 | 72/98=73.5% | layer3.40-59 + layer4 |

这保持了相同的解冻策略：从网络的约 73.5% 深度位置开始解冻后部，实现渐进式微调。

**为什么选择 body.72？**
- 保持与 IR-100 相同的解冻比例（73.5%）
- 兼容现有的 `body.X` 命名体系
- 解冻 53.66% 的模型参数（约 6380 万参数）

## 训练脚本

### 脚本位置
```bash
/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/scripts/train/train_topofr_ir200_0605.sh
```

### 预训练模型
```bash
/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200/Glint360K_R200_TopoFR_9784.pt
```

- 大小: 1.13 GB
- 包含 1808 个参数张量
- 在 Glint360K 数据集上预训练
- 验证准确率: 97.84%

### 四阶段训练流程

#### 阶段 1: PartialFC Warmup（5 epochs）
- **目标**: 在冻结 backbone 的情况下预热分类器
- **冻结**: 整个 backbone
- **训练**: PartialFC 分类器
- **学习率**: 0.008
- **数据增强**: basic_v2_numpy

#### 阶段 2: 部分 Backbone 解冻（15 epochs）
- **目标**: 微调网络后部以适应新数据
- **冻结**: body.0-71（layer3.0-39） + PartialFC 分类器
- **训练**: body.72-97（layer3.40-59 + layer4） + output layers
- **解冻参数**: 53.66%（约 6380 万参数）
- **学习率**: 0.008 (cosine decay)
- **数据增强**: gridsample_v2_numpy
- **Warmup**: 2 epochs

#### 阶段 3: PartialFC 重训练（5 epochs）
- **目标**: 在更新后的 backbone 基础上重新训练分类器
- **冻结**: 整个 backbone
- **训练**: PartialFC 分类器
- **学习率**: 0.006

#### 阶段 4: 全模型微调（15 epochs）
- **目标**: 解冻整个 backbone 进行端到端微调
- **冻结**: PartialFC 分类器
- **训练**: 整个 backbone
- **学习率**: 0.0008 (cosine decay)
- **Warmup**: 2 epochs

### 训练参数

```bash
GPU: 8 卡 (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
Batch Size: 128 per GPU (总计 1024)
Precision: bf16-mixed
Workers: 8 per GPU
PartialFC Sample Rate: 0.40
Loss: TopoFR + AdaFace
```

## 使用方法

### 基本用法
```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr
bash scripts/train/train_topofr_ir200_0605.sh
```

### 自定义配置
通过环境变量覆盖默认配置：

```bash
# 自定义数据目录
DATA_ROOT=/path/to/data bash scripts/train/train_topofr_ir200_0605.sh

# 自定义预训练模型
PRETRAINED_MODEL=/path/to/model.pt bash scripts/train/train_topofr_ir200_0605.sh

# 自定义输出目录
OUTPUT_ROOT=/path/to/output bash scripts/train/train_topofr_ir200_0605.sh

# 自定义阶段前缀
STAGE1_PREFIX=my_stage1 bash scripts/train/train_topofr_ir200_0605.sh

# 启用最终评估
SKIP_FINAL_EVAL=False bash scripts/train/train_topofr_ir200_0605.sh
```

### 断点续训
脚本自动支持断点续训：
- 每个阶段会自动搜索已完成或未完成的 checkpoint
- 已完成的阶段会自动跳过并复用结果
- 中断的阶段会自动从最新 checkpoint 继续训练
- 无需手动指定 checkpoint 路径

### 单独运行某个阶段
通过覆盖前缀，可以单独运行某个阶段：

```bash
# 只运行阶段 1
STAGE2_PREFIX=skip_stage2 \
STAGE3_PREFIX=skip_stage3 \
STAGE4_PREFIX=skip_stage4 \
bash scripts/train/train_topofr_ir200_0605.sh
```

## 测试脚本

验证配置是否正确：

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr
bash scripts/test_ir200_config.sh
```

测试内容：
- ✓ 预训练模型文件存在性
- ✓ 权重加载
- ✓ 模型结构分析（layer blocks 统计）
- ✓ IR-200 模型初始化
- ✓ 预训练权重加载到模型
- ✓ 前向传播测试

## 预期训练时间（估算）

基于 8 x A100/H100 GPUs，batch_size=128：

- **阶段 1**: ~2-3 小时（5 epochs）
- **阶段 2**: ~6-8 小时（15 epochs）
- **阶段 3**: ~2-3 小时（5 epochs）
- **阶段 4**: ~6-8 小时（15 epochs）
- **总计**: ~16-22 小时

注意：IR-200 比 IR-100 慢约 1.8-2.0 倍。

## 输出结构

```
${OUTPUT_ROOT}/
├── s1_topofr_ir200_warmup_0605_Sep-08_001/
│   ├── checkpoints_every_epoch/
│   │   ├── epoch:0_step:xxxx/
│   │   │   ├── model.pt
│   │   │   └── pipeline.pt
│   │   └── epoch:4_step:xxxx/
│   └── logs/
├── s2_topofr_ir200_layer348_0605_Sep-08_001/
├── s3_topofr_ir200_classifier_0605_Sep-08_001/
└── s4_topofr_ir200_full_0605_Sep-08_001/
```

## 与 IR-100 的主要区别

| 项目 | IR-100 | IR-200 |
|------|--------|--------|
| 配置文件 | `v1_ir101.yaml` | `v1_ir200.yaml` |
| 模型名称 | `ir101` | `ir200` |
| Layer3 blocks | 30 | 60 |
| 阶段2解冻点 | `body.36` | `layer3.48` |
| 训练时间 | 基准 | ~2倍 |
| 显存占用 | 基准 | ~1.8倍 |
| 预训练模型 | Glint360K_R100_TopoFR | Glint360K_R200_TopoFR |

## 注意事项

1. **显存需求**: IR-200 需要更多显存，确保每张卡至少有 24GB（推荐 40GB+）
2. **预训练权重**: 注意 checkpoint 中有一个 `weight` 键（分类器头），加载时会显示为 unexpected key，这是正常的
3. **BN 统计**: 阶段 2 使用 `models.freeze=True` 确保只有解冻部分的 BN 更新统计量
4. **解冻策略**: body.72 表示从第 72 个 block 开始解冻（对应 layer3.40），保持与 IR-100 相同的 73.5% 解冻比例
5. **PartialFC**: sample_rate=0.40 表示每个 batch 只采样 40% 的类别进行训练

## 故障排查

### 模型加载失败
```bash
# 检查预训练模型
ls -lh /root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200/

# 运行测试脚本
bash scripts/test_ir200_config.sh
```

### OOM 错误
- 减小 batch_size（当前 128）
- 使用更少的 GPU
- 确认使用 bf16-mixed precision

### 找不到 ir200 配置
```bash
# 检查配置文件
ls -la models/iresnet_insightface/configs/v1_ir200.yaml

# 检查模型加载器
grep -n "ir200" models/iresnet_insightface/__init__.py
```

### KeyError: 'body.72' 或 'layer3.X'
这通常意味着 `pefts/__init__.py` 没有正确扩展以支持 IR-200。确保：
- IR-200 的 98 个 blocks 映射已添加（body.0-97）
- layer 访问支持已添加
- 测试通过：`bash scripts/test_body72_ir200.sh`

## 相关文件

- **训练脚本**: `scripts/train/train_topofr_ir200_0605.sh`
- **测试脚本**: `scripts/test_ir200_config.sh`
- **模型配置**: `models/iresnet_insightface/configs/v1_ir200.yaml`
- **模型定义**: `models/iresnet_insightface/model.py`
- **模型加载**: `models/iresnet_insightface/__init__.py`
- **预训练权重**: `/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr200/Glint360K_R200_TopoFR_9784.pt`

## 参考

基于 IR-100 训练脚本改进：
- 原始脚本: `scripts/train/train_topofr_0605.sh`
- TopoFR 论文中的四阶段训练策略
- Glint360K 数据集预训练

# TopoFR Integration Guide

## 概述

已将 TopoFR (NeurIPS 2024) 的拓扑对齐损失集成到 CVLface 训练框架中。

## 核心组件

### 1. 损失函数
- `losses/topology.py` - 拓扑损失计算（持久同调）
- `losses/gum.py` - GUM模型（样本加权）
- `losses/topofr_loss.py` - TopoFR组合损失
- `losses/configs/topofr.yaml` - TopoFR配置文件

### 2. Pipeline
- `pipelines/train_model_cls_topofr_pipeline.py` - TopoFR训练pipeline

### 3. Classifier修改
- `classifiers/partial_fc/partial_fc.py` - 支持TopoFR的PartialFC
- `classifiers/fc/fc.py` - 支持TopoFR的FC

## 使用方法

### 配置文件

使用TopoFR损失训练时，修改loss配置：

```yaml
# losses=configs/topofr.yaml
margin_loss_name: 'topofr'
base_loss: 'arcface'  # 或 'cosface', 'adaface'
s: 64.0
m: 0.5
topo_weight: 0.1  # 拓扑损失权重（论文中使用0.1）
use_gum: true     # 是否使用GUM模型进行样本加权
temp: 1           # GUM温度参数
```

### 训练命令示例

使用原有的 `train_opt.py`，只需更改loss配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=topofr_train_0605 \
    trainers.num_gpu=8 trainers.batch_size=256 trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    models=iresnet/configs/v1_ir101.yaml \
    models.start_from=/path/to/pretrained/model.pt \
    dataset=configs/dataset_0605_train_rec.yaml \
    data_augs=configs/basic_v2_numpy.yaml \
    classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.40 \
    losses=configs/topofr.yaml \
    evaluations=configs/val_20260605.yaml \
    optims=configs/step_sgd.yaml \
    optims.lr=0.001 optims.num_epoch=20 \
    optims.momentum=0.9 optims.weight_decay=0.0005 \
    optims.max_grad_norm=5.0 \
    dataset.model_save_dir=/data1/dataset_0605/train_output
```

### 多阶段训练示例

按照原有的多阶段训练流程，只需在loss配置中切换为TopoFR：

```bash
# 阶段1: 分类器warmup (使用TopoFR)
fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=s1_topofr_warmup_0605 \
    models.freeze=True \
    pefts=configs/freeze.yaml \
    losses=configs/topofr.yaml \
    # ... 其他参数

# 阶段2: body.36解冻训练
fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=s2_topofr_body36_0605 \
    models.freeze=False \
    pefts=configs/part_freeze.yaml pefts.target_modules=body.36 \
    losses=configs/topofr.yaml \
    # ... 其他参数

# 阶段3: 分类器重训练
# 阶段4: 全模型微调
# 均可使用 losses=configs/topofr.yaml
```

## TopoFR vs 其他损失的区别

### 标准训练（AdaFace）
```yaml
losses=configs/adaface.yaml
# 仅使用分类损失 + AdaFace自适应margin
```

### TopoFR训练
```yaml
losses=configs/topofr.yaml
base_loss: 'adaface'  # 基础损失选择AdaFace
topo_weight: 0.1      # 添加拓扑损失
use_gum: true         # 添加GUM样本加权
```

**组合损失公式**:
```
total_loss = weighted_classification_loss + 0.1 * topological_loss

其中:
- weighted_classification_loss 使用GUM模型和预测置信度动态加权
- topological_loss 保持输入空间和特征空间的拓扑结构一致性
```

## 配置文件

提供了三种TopoFR配置：

1. **topofr.yaml** - 基于ArcFace
2. **topofr_cosface.yaml** - 基于CosFace（论文中标记为†）
3. **topofr_adaface.yaml** - 基于AdaFace（推荐）

## 性能优化

TopoFR增加的计算开销：
- **拓扑损失计算**: 需要计算距离矩阵和持久同调（约10-15%训练时间）
- **GUM模型**: EM算法样本加权（约5%训练时间）

**总开销**: 约15-20%额外训练时间

**优化建议**:
- 使用较大的batch size以摊销拓扑损失开销
- 可以设置 `topo_weight: 0` 暂时关闭拓扑损失进行快速实验
- 可以设置 `use_gum: false` 关闭GUM加权

## 论文结果对比

原始TopoFR论文结果（IJB-C）：

| 训练数据 | 模型 | 基础Loss | IJB-C(1e-5) | IJB-C(1e-4) |
|---------|------|---------|-------------|-------------|
| MS1MV2 | R50 | CosFace† | 94.79% | 96.42% |
| MS1MV2 | R50 | ArcFace | 94.71% | 96.49% |
| Glint360K | R100 | ArcFace | 96.57% | 97.60% |

† 表示使用CosFace作为base_loss

## 故障排查

### 问题1: 导入错误
```
ImportError: cannot import name 'TopoFRLoss'
```
**解决**: 确保新文件已创建并且在losses/__init__.py中正确导入

### 问题2: 内存不足
拓扑损失需要计算距离矩阵，占用O(batch_size^2)内存

**解决**: 
- 减小batch_size
- 使用梯度累积
- 或暂时设置 `topo_weight: 0`

### 问题3: 训练速度慢
**解决**:
- 确保使用turbojpeg解码后端
- 使用persistent_workers=True
- 考虑减小验证频率

## 参考

论文: [TopoFR: A Closer Look at Topology Alignment on Face Recognition](https://proceedings.neurips.cc/paper_files/paper/2024/hash/419b6c974712adb884bfbbeea8e94d1b-Abstract-Conference.html)

原始代码: https://github.com/modelscope/facechain/tree/main/face_module/TopoFR

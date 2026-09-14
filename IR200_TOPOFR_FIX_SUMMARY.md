# IR-200 TopoFR 训练适配 - 问题修复总结

## 问题描述

在第二阶段训练时遇到错误：
```
KeyError: 'layer3.48'
```

## 根本原因

原始代码的 `part_freeze` 功能只支持 IR-100 的 49 个 blocks（`body.0` 到 `body.48`），不支持：
1. IR-200 的 98 个 blocks
2. 直接使用 `layer3.X` 这样的命名方式

## 解决方案

### 1. 扩展 `pefts/__init__.py`

添加了对 IR-200 的完整支持：

**修改前**：
- 只支持 IR-100 的 49 个 blocks
- 使用硬编码的 layer 大小（3-13-30-3）

**修改后**：
- 支持 IR-100 的 49 个 blocks（body.0-body.48）
- 支持 IR-200 的 98 个 blocks（body.0-body.97）
- 添加直接 layer 访问（layer1.X, layer2.X, layer3.X, layer4.X）
- 对 IR-100 和 IR-200 都生成正确的参数映射

### 2. 修正解冻点选择

**初始错误方案**：
- 使用 `layer3.48`（不在预定义映射中）
- 计算逻辑：layer3 的 80% = 60 × 0.8 = 48

**最终正确方案**：
- 使用 `body.72`（在扩展的映射中）
- 计算逻辑：保持与 IR-100 的 `body.36` 相同比例
  - IR-100: body.36 / 49 = 73.5%
  - IR-200: 98 × 0.735 ≈ 72 → `body.72`
  - body.72 = layer1[6] + layer2[26] + layer3[40] = 从 layer3.40 开始解冻

## 架构对应关系

### IR-100 (49 blocks)
```
body.0-2   → layer1.0-2   (3 blocks)
body.3-15  → layer2.0-12  (13 blocks)
body.16-45 → layer3.0-29  (30 blocks)
body.46-48 → layer4.0-2   (3 blocks)
```

**body.36**:
- 索引：36 = 3 + 13 + 20
- 位置：layer3.20（layer3 的第 20 个 block）
- 比例：36/49 = 73.5%
- 解冻：layer3.20-29 + layer4.0-2 + output layers

### IR-200 (98 blocks)
```
body.0-5   → layer1.0-5   (6 blocks)
body.6-31  → layer2.0-25  (26 blocks)
body.32-91 → layer3.0-59  (60 blocks)
body.92-97 → layer4.0-5   (6 blocks)
```

**body.72**:
- 索引：72 = 6 + 26 + 40
- 位置：layer3.40（layer3 的第 40 个 block）
- 比例：72/98 = 73.5%（与 IR-100 相同）
- 解冻：layer3.40-59 + layer4.0-5 + output layers

## 验证结果

### ✅ 配置测试通过

```bash
$ bash scripts/test_body72_ir200.sh

✓ IR-200 模型初始化成功
✓ 总参数量: 118,833,920
✓ part_freeze 应用成功

参数统计:
  可训练参数: 63,761,408 (53.66%)
  冻结参数: 55,072,512 (46.34%)

关键层验证:
  ✓ net.layer3.39: 冻结 (应该被冻结)
  ✓ net.layer3.40: 可训练 (应该被解冻)
  ✓ net.layer3.59: 可训练 (应该被解冻)
  ✓ net.layer4.0: 可训练 (应该被解冻)
  ✓ net.bn2: 可训练 (应该被解冻)
  ✓ net.fc: 可训练 (应该被解冻)
```

### 解冻参数分析

**解冻部分（53.66% 参数）**：
- layer3.40 到 layer3.59（20 个 blocks）
- layer4.0 到 layer4.5（6 个 blocks）
- bn2, fc, features（output layers）
- 共约 6380 万参数

**冻结部分（46.34% 参数）**：
- layer1.0 到 layer1.5（6 个 blocks）
- layer2.0 到 layer2.25（26 个 blocks）
- layer3.0 到 layer3.39（40 个 blocks）
- 共约 5507 万参数

## 更新的文件

### 1. 核心代码修改
```
✓ pefts/__init__.py
  - 添加 IR-100 完整映射（49 blocks）
  - 添加 IR-200 完整映射（98 blocks）
  - 添加直接 layer 访问支持
  - 修复参数匹配逻辑
```

### 2. 训练脚本更新
```
✓ scripts/train/train_topofr_ir200_0605.sh
  - 阶段2: layer3.48 → body.72
  - 前缀: s2_topofr_ir200_layer348_0605 → s2_topofr_ir200_body72_0605
  - 注释: 更新解冻点说明
```

### 3. 测试脚本
```
✓ scripts/test_body72_ir200.sh (新建)
  - 验证 body.72 配置
  - 检查参数训练状态
  - 确认解冻边界正确
```

## 设计决策说明

### 为什么选择 body.72 而不是 layer3.48？

1. **兼容性**：`body.X` 是 TopoFR 代码中的标准命名，IR-100 训练已经使用
2. **比例一致**：body.72/98 = body.36/49 ≈ 73.5%，保持相同的训练策略
3. **代码简洁**：利用现有的 body 映射，不需要引入新的命名体系
4. **可预测性**：用户可以清楚地看到解冻比例（72/98）

### 为什么不是 body.71 或 body.73？

- body.71 = 6 + 26 + 39 → layer3.39（72.4%）
- **body.72 = 6 + 26 + 40 → layer3.40（73.5%）** ✓ 最接近 IR-100 的比例
- body.73 = 6 + 26 + 41 → layer3.41（74.5%）

## 当前训练配置

### 阶段 1: PartialFC Warmup
- **解冻**: 无（全部冻结）
- **训练**: PartialFC 分类器
- **配置**: `pefts=configs/freeze.yaml`

### 阶段 2: 部分 Backbone 解冻
- **解冻**: body.72-97 (layer3.40-59 + layer4.0-5 + outputs)
- **训练**: 后部 backbone（53.66% 参数）
- **配置**: `pefts=configs/part_freeze.yaml pefts.target_modules=body.72`

### 阶段 3: PartialFC 重训练
- **解冻**: 无（全部冻结）
- **训练**: PartialFC 分类器
- **配置**: `pefts=configs/freeze.yaml`

### 阶段 4: 全模型微调
- **解冻**: 全部 backbone
- **训练**: 整个 backbone（分类器冻结）
- **配置**: `pefts=configs/full.yaml`

## 下一步

训练脚本已修复，可以继续运行：

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr
bash scripts/train/train_topofr_ir200_0605.sh
```

或使用快速启动：

```bash
bash scripts/train/quick_start_ir200.sh
```

如果阶段1已经完成，脚本会自动从阶段2继续。

## 相关测试脚本

```bash
# 测试基础配置和模型加载
bash scripts/test_ir200_config.sh

# 测试 body.72 解冻配置
bash scripts/test_body72_ir200.sh
```

---

**修复状态**: ✅ 完成
**测试状态**: ✅ 通过
**准备训练**: ✅ 就绪

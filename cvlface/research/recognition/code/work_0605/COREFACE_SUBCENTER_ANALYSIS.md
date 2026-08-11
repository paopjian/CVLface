# CoreFace + SubCenter 组合实验分析

更新日期：2026-08-10

## 1. 结论摘要

当前 `coreface_subcenter` 实验表现出明显的 SubCenter 特征，但没有保留旧 CoreFace 在 1201、3T、Enhance 等测试集上的优势。主要原因不是 CoreFace 没有启用，而是：

1. SubCenter 和 CoreFace 都会改变 embedding 的几何结构，但方向不完全一致。
2. K=3 SubCenter 使用硬 `max` 选择子中心，两个 dropout view 可能选择不同的子中心；CoreFace 又要求两个 view 保持接近，产生路由和梯度冲突。
3. 当前 CoreFace 对比项权重很小，只是弱正则，无法抵消 K=3 分类目标。
4. 组合实验同时改变了 batch、增强、学习率和 epoch，不能把退化全部归因于 SubCenter 实现。
5. CoreFace 的对比损失只使用单卡本地 batch，而分类器会跨卡聚合；S4 每卡 batch 从 256 降到 128，专门削弱了 CoreFace 的负样本池。

因此，当前结果更准确的描述是：**K=3 分类目标主导了训练，CoreFace 只提供了较弱且与子中心路由不完全一致的正则，最终形成了带额外优化噪声的 SubCenter 风格模型。**

这不是证明两种方法天然不能结合，而是说明“全程 K=3 + 低权重 CoreFace + 硬子中心路由”的直接叠加方式不理想。

## 2. 实验对象和数据来源

### 2.1 线上分支

- 分支：`origin/coreface_subcenter`
- 组合训练 run：`try_coreface_subcenter/sedie2xi`
- run 名称：`coreface_subcenter_s4_joint_after_s2_sgd20_0605_08-06_0`
- 独立测试 run：`work_0605_test/4jsc6knz`
- run 名称：`coreface_subcenter_s4`

训练期 run 只用于公开集和 TinyFace；独立测试 run 用于 1201、3T、Glint、Enhance、IJBC 和 IJBC-001。两类 run 中可能存在同名指标，不能混用。

### 2.2 评估现象

CoreFace-SubCenter 最佳综合 epoch 为 18，分项如下：

| 开源 | TinyFace | IJBC | 1201 | 3T | Glint | Enhance |
|---:|---:|---:|---:|---:|---:|---:|
| 96.228 | 73.203 | 75.755 | 72.763 | 93.424 | 78.345 | 52.577 |

相对旧 CoreFace-SGD20 e14：

| 指标 | 变化 |
|---|---:|
| 开源 | +0.189 |
| TinyFace | +2.066 |
| IJBC | -0.052 |
| 1201 | -0.832 |
| 3T | -0.903 |
| Glint | +0.660 |
| Enhance | -3.238 |

这说明组合模型确实继承了 SubCenter 的公开集、TinyFace 和极低 FAR 特征，但没有继承旧 CoreFace 的跨域稳定性。

## 3. 两种方法的目标差异

### 3.1 SubCenter

SubCenter 不是一个独立的 backbone loss，而是分类器的代理中心结构：每个身份维护 K 个中心，类别 logit 取这些中心的最大相似度。K=3 的目的，是允许一个身份存在多个外观模式，从而放松类内紧凑性并提高噪声鲁棒性。

线上实现位于：

`try_coreface_subcenter/classifiers/partial_fc/partial_fc.py`

其中 `compute_logits()` 在 `num_subcenters > 1` 时执行：

```python
logits.reshape(batch, classes, num_subcenters).amax(dim=2)
```

`amax` 的梯度只会传给当前胜出的子中心。一个样本每次更新的通常是一个中心，而不是该身份的全部中心。

### 3.2 CoreFace

CoreFace 使用同一张图像经过两次 dropout 得到 `feat1` 和 `feat2`，然后：

- 两个 view 分别经过分类器；
- 对比损失要求同一图像的两个 view 相近；
- 对其他身份构造困难负样本；
- 同一身份的其他图像不作为负样本。

线上 pipeline 的核心代码位于：

`try_coreface_subcenter/pipelines/train_model_cls_pipeline.py`

组合损失为：

```text
0.5 * classification(view1)
+ 0.5 * classification(view2)
+ 0.05 * contrast(view1, view2)
+ 0.0 * contrast(view2, view1)
```

配置位于：

`try_coreface_subcenter/pipelines/configs/train_model_cls_coreface.yaml`

### 3.3 为什么会发生冲突

同一图像的两个 dropout view 经过扰动后，可能分别落在同一身份的不同子中心一侧：

```text
原图
  ├─ view1 -> 子中心 A，分类梯度指向 A
  └─ view2 -> 子中心 B，分类梯度指向 B

CoreFace 对比项 -> 要求 view1 和 view2 接近
```

因此冲突点不是“身份类别不可区分”，而是“同一身份应该使用哪个子中心”的路由不稳定。特征接近子中心边界时，dropout、GridSample 或其他扰动都可能改变 `amax` 的胜者，造成：

- 两个 view 的分类梯度方向不一致；
- 非胜出子中心没有梯度；
- 子中心分配在 batch 之间抖动；
- CoreFace 的一致性梯度与 SubCenter 的多模态梯度相互抵消。

普通 K=1 分类器没有这个路由问题：两个 view 最终都指向同一个身份中心，因此 CoreFace 更容易发挥正则作用。

## 4. 当前实现中已确认的弱点

### 4.1 CoreFace 权重过小

配置中 `coreface_weight_contrast=0.05`，而两个分类项权重合计为 `1.0`。W&B 组合 run 的 e8-e19 训练记录中：

- 分类 view loss 大约为 `3.69-3.96`；
- contrast loss 大约为 `1.29`；
- 对比项的标量加权贡献约为 `0.065`。

实际梯度量不能仅由 loss 标量判断，但可以确认 CoreFace 在优化目标中是弱正则，不是与分类器等强度的共同目标。

### 4.2 CoreFace 对比损失没有跨卡负样本

pipeline 先在当前 rank 上调用 CoreFace 对比损失，再进入 PartialFC。PartialFC 的 `AllGather` 只发生在分类器内部。

因此：

- 分类损失使用全局 batch 的 embedding；
- CoreFace 只看到当前 GPU 的本地 batch；
- S4 每卡 batch 从旧 CoreFace 的 256 改为 128；
- 每个 anchor 的候选负样本数量从约 255 降到约 127。

这会专门降低 CoreFace 的 hard-negative 质量。不能用全局 batch size 代替 CoreFace 实际看到的本地 batch size。

### 4.3 组合实验不是单变量实验

旧 CoreFace S4 脚本（`origin/coreface`）与线上组合分支的关键差异：

| 项目 | 旧 CoreFace | CoreFace-SubCenter |
|---|---:|---:|
| 分类器 | K=1 PFC | K=3 PFC |
| S4 每卡 batch | 256 | 128 |
| S4 增强 | Basic | GridSample |
| backbone LR | 0.0008 | 0.0004 |
| classifier LR | 0.00005 | 0.000025 |
| S4 epoch | 15 | 20 |

所以目前不能得出“只要把 CoreFace 和 SubCenter 结合就会退化”的严格结论。至少需要控制其他变量后再判断。

### 4.4 SubCenter 原始流程没有完成最后的中心收缩

原始 SubCenter ArcFace 的典型流程是：

1. 使用 K=3 训练，发现类内多中心和噪声；
2. 删除非主中心以及高置信噪声样本；
3. 转回普通 ArcFace/K=1 做最终训练。

当前组合分支从 S1 到 S4 一直保留 K=3，没有把子中心阶段的结果转成单中心。因此它会持续保留“类内多簇”结构，与 CoreFace 希望最终加强的一致性存在结构性张力。

## 5. 尚未确认、需要消融验证的问题

以下是合理假设，但不能仅凭当前 W&B 结果确认：

1. `amax` 路由切换的比例是否显著升高。需要记录两个 view 对每个身份选择的子中心编号，以及 view1/view2 选择不同编号的比例。
2. 非胜出子中心是否长期不更新。需要统计每个子中心的激活次数、梯度次数和中心范数。
3. CoreFace 对比项是否因本地 batch 过小而失去 hard-negative 效果。需要保持每卡 batch=256，或者改为跨卡 gather 后再计算对比矩阵。
4. GridSample 是否单独造成了 1201、3T、Enhance 退化。需要用 Basic 和 GridSample 做 S4 配对实验。
5. 组合模型退化主要来自 K=3，还是来自 S4 学习率减半。需要恢复旧 CoreFace 的 `0.0008/0.00005` 做控制实验。

## 6. 推荐的最小消融顺序

### 实验 A：先验证训练配方

保持 K=1，只把旧 CoreFace 的 S4 配方复制到当前代码，使用：

```text
每卡 batch=256
Basic augmentation
backbone lr=0.0008
classifier lr=0.00005
S4=15 epoch
```

如果 K=1 也退化，问题主要在增强、batch 或优化器配方，而不是 SubCenter。

### 实验 B：只替换分类器

固定实验 A 的所有参数，只改 `num_subcenters=3`。这才是判断“CoreFace + SubCenter”本身是否冲突的有效对照。

### 实验 C：SubCenter 预训练后转 K=1

推荐的实际组合方式：

```text
S1/S2: K=3 SubCenter，学习多中心/处理噪声
中心统计: 为每类选择主子中心
S3: 将主子中心迁移到 K=1 分类器
S4: 使用 K=1 + CoreFace 联合训练
```

这相当于让 SubCenter 负责鲁棒初始化，让 CoreFace 负责最终特征收紧，避免两个目标在最终阶段同时竞争。

### 实验 D：若必须保留 K=3

可以进一步尝试：

- 使用 softmax 加权子中心，而不是硬 `amax`；
- 让两个 view 共享同一个子中心路由；
- 逐步增加 CoreFace 权重，而不是固定 `0.05`；
- 将对比损失改为跨卡 gather；
- 记录并监控子中心路由一致率。

这些属于后续研究方案，不应与基础配方控制实验同时修改。

## 7. 最终判断

当前结果支持以下判断：

- **CoreFace 并非没有运行**：组合 run 从 epoch 8 开始记录 `coreface_active=1`，并且有非零 contrast loss。
- **SubCenter 确实主导了最终 embedding 几何**：公开集、TinyFace、Glint 极低 FAR 的变化方向与独立 SubCenter 一致。
- **当前组合出现了目标不匹配和训练配方混杂**：K=3 硬路由、弱对比权重、本地负样本池减小、GridSample 和 S4 超参数同时变化。
- **不能据此断言两种方法原则上不能结合**：需要先做 K=1/K=3 单变量对照；从工程和训练稳定性看，SubCenter 预训练后转 K=1 再接 CoreFace 更合理。

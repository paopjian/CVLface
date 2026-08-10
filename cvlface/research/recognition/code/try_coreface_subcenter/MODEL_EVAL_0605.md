# 0605 模型评估与 Checkpoint 记录

初始记录日期：2026-08-04；新增模型更新：2026-08-10

本文沉淀 0605 实验中 AdaFace、QGFace6、CoreFace-SGD20、SubCenter 以及新增 CoreFace-SubCenter-S4 的横向评估结论、可用 checkpoint 与推理约定。它用于后续实验和对话的上下文，不能替代从 W&B 原始记录重新计算的结果。

## 结论适用范围

- 综合排名使用每个模型在全部已评估 epoch 中的最佳 checkpoint，而不是统一固定 epoch。
- 当前综合分**包含 Enhance**。若排除 Enhance、调整测试集权重或调整 FAR 档位，必须按下述口径重新计算，不能沿用本文排名。
- 训练期公开集指标与独立评估 run 的私有集指标必须分开取值；两类 run 可能出现同名指标，不能混用。
- 本文所有 epoch 与对应 checkpoint 目录保持一致，例如 `epoch:14`。

## 评分口径

```text
开源分 = mean(AgeDB, CALFW, CPLFW)
IJBC-001 = mean(FAR@1e-5, 1e-6, 5e-7, 1e-7)
IJBC分 = mean(官方 IJBC, IJBC-001)

综合分 =
    开源分 x 10%
  + TinyFace Rank-1 x 10%
  + IJBC分 x 20%
  + 1201 x 15%
  + 3T x 15%
  + Glint x 15%
  + Enhance x 15%
```

其中 1201、3T、Glint、Enhance 分别取 FAR `1e-6` 至 `1e-10` 五档指标的均值。该口径下 QGFace6 比 CoreFace-SGD20 高 `0.219442` 分。

## 最优 Checkpoint

| 模型 | 最优 epoch | 综合分 | 次优 epoch | 次优综合分 |
| --- | ---: | ---: | ---: | ---: |
| QGFace6 | 9 | 77.311813 | 0 | 77.260480 |
| CoreFace-SGD20 | 14 | 77.092370 | 13 | 77.081334 |
| CoreFace-SubCenter-S4 | 18 | 76.660443 | 19 | 76.619094 |
| SubCenter | 14 | 75.966277 | 12 | 75.929335 |
| AdaFace | 12 | 75.440411 | 9 | 75.395816 |

CoreFace 的 epoch 12 综合分为 `77.081302`，与 epoch 13 和 14 非常接近。

### 新增 CoreFace-SubCenter-S4

新增组合使用两个不同来源的 run 合并同一 epoch：

- 训练期公开集与 TinyFace：`try_coreface_subcenter/sedie2xi`，run 名称 `coreface_subcenter_s4_joint_after_s2_sgd20_0605_08-06_0`。
- 独立测试集：`work_0605_test/4jsc6knz`，run 名称 `coreface_subcenter_s4`。

按每个模型选择自己的最优 epoch，新增模型最佳为 **epoch 18，综合分 `76.660443`**。逐 epoch 综合分如下：

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 综合分 | 73.198 | 74.021 | 74.490 | 74.698 | 75.210 | 75.312 | 75.168 | 75.444 | 75.667 | 75.894 |

| epoch | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | **18** | 19 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 综合分 | 75.995 | 76.017 | 76.165 | 76.234 | 76.276 | 76.537 | 76.551 | 76.595 | **76.660** | 76.619 |

最佳 epoch 18 的分项结果：

| 开源 | TinyFace | IJBC | 1201 | 3T | Glint | Enhance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96.228 | 73.203 | 75.755 | 72.763 | 93.424 | 78.345 | 52.577 |

与旧 CoreFace-SGD20 e14 的差值为：开源 `+0.189`、TinyFace `+2.066`、IJBC `-0.052`、1201 `-0.832`、3T `-0.903`、Glint `+0.660`、Enhance `-3.238`，综合分由 `77.092370` 降至 `76.660443`（`-0.431927`）。因此它确实改善了公开识别、TinyFace 和 Glint，但没有解决 CoreFace 的跨私有集稳定性问题，Enhance 退化抵消了公开集收益。

Glint 均值的提升主要来自 FAR `1e-10` 由旧 CoreFace 的 `34.397` 提升到 `40.941`（`+6.544`）；FAR `1e-6` 至 `1e-9` 分别下降 `0.471`、`0.884`、`1.318`、`0.567`。该变化符合 SubCenter 强化极低 FAR 的特点，但不是 Glint 全档位的普遍提升。1201 五档全部下降；3T 的下降随 FAR 降低而扩大，FAR `1e-10` 下降 `2.310`；Enhance 五档下降约 `2.4` 至 `3.8`。

将新增模型作为第五个候选时，当前排名为：QGFace6 e9 (`77.312`) > CoreFace-SGD20 e14 (`77.092`) > CoreFace-SubCenter-S4 e18 (`76.660`) > SubCenter e14 (`75.966`) > AdaFace e12 (`75.440`)。

| 模型 checkpoint | 开源 | TinyFace | IJBC | 1201 | 3T | Glint | Enhance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QGFace6 e9 | 95.833 | 72.881 | 76.007 | 74.288 | 94.281 | 68.187 | 64.838 |
| CoreFace-SGD20 e14 | 96.039 | 71.137 | 75.807 | 73.595 | 94.327 | 77.685 | 55.815 |
| CoreFace-SubCenter-S4 e18 | 96.228 | 73.203 | 75.755 | 72.763 | 93.424 | 78.345 | 52.577 |
| SubCenter e14 | 96.100 | 72.961 | 75.898 | 72.963 | 93.627 | 78.591 | 47.356 |
| AdaFace e12 | 96.172 | 73.069 | 75.331 | 73.354 | 93.922 | 74.076 | 48.316 |

## 各模型特点

### QGFace6 e9

- 综合第一；IJBC-001 的 FAR `1e-5`、`1e-6`、`5e-7` 最强，1201 前四级 FAR 第一，Enhance 五级 FAR 均第一。
- 优势集中在困难集与低 FAR。
- Glint 是明确短板，五级 FAR 均为四者最后，FAR `1e-10` 为 `4.064`，远低于 SubCenter 的 `41.982`；CPLFW 也是四者最低，官方 IJBC 略低。

### CoreFace-SGD20 e14

- 综合第二，仅落后 QGFace6 `0.219442` 分，是四者中最均衡的选择。
- 官方 IJBC 第一；3T 前四级 FAR 第一；Glint 前四级 FAR 第一；1201 与 Enhance 通常稳定在第二梯队。
- TinyFace Rank-1 为 `71.137`，比其余三者低约 `1.7` 至 `1.9` 个百分点，是最明显短板。IJBC-001 极低 FAR 不如 QGFace6。

### CoreFace-SubCenter-S4 e18

- 开源分与 TinyFace 在五个候选中第一，明显修复了旧 CoreFace 的 TinyFace 短板；Glint 五档均值第二，仅略低于 SubCenter。
- Glint 优势集中在 FAR `1e-10`，前四档仍低于旧 CoreFace，说明 SubCenter 的极低 FAR 特性被保留下来。
- 1201 和 3T 在五个候选中最低，Enhance 低于 QGFace6 和旧 CoreFace；当前权重下不如直接使用旧 CoreFace-SGD20。

### SubCenter e14

- 综合第四；官方 IJBC 接近 CoreFace，公开集表现强，Glint 五档均值第一。
- Glint 的优势主要来自 FAR `1e-10 = 41.982`；前四级 FAR 实际由 CoreFace 领先。
- Enhance 最低，3T 五级 FAR 都最后，1201 整体偏低，跨域稳定性不及 QGFace6 与 CoreFace。

### AdaFace e12

- 综合第五；在原四模型中开源集和 TinyFace 均为第一，但已被新增 CoreFace-SubCenter-S4 超过，常规公开人脸识别能力仍然较强。
- Glint 居中，官方 IJBC 与第一名接近。
- 困难私有集是主要短板：IJBC-001 的 FAR `1e-6`、`5e-7` 最低，Enhance 明显低于 QGFace6 与 CoreFace，1201 和 3T 无明显优势。

## W&B 取数来源

报告：<https://wandb.ai/kejian-zhao-tsinghua-university/work_0605_test/reports/---VmlldzoxNzYyMDQzMQ>

训练期公开测试 run，只用于 TinyFace、AgeDB、CALFW、CPLFW：

| 模型 | W&B 项目 | Run 名称 | Run ID |
| --- | --- | --- | --- |
| SubCenter | `try_subcenter` | `subcenter_s4_joint04_0605_07-22_1` | `6j794vzl` |
| CoreFace | `try_coreface` | `coreface_s4_joint_after_s2_sgd20_0605_08-01_0` | `4oamvp70` |
| QGFace6 | `qgface` | `qgface_6_ir101_epoch11_model_realign_07-18_1` | `qlu81xsn` |
| AdaFace | `work_0605` | `s4_full_0605_06-11_0` | `6itzqdk6` |

独立评估 run，用于 1201、3T、Glint、Enhance、IJBC 与 IJBC-001：

| 模型 | Run 名称 | Run ID |
| --- | --- | --- |
| AdaFace | `s4_full_0605_trt_v1` | `9akvergx` |
| QGFace6 | `qgface_6_enhance` | `mwypg43o` |
| SubCenter | `subcenter_s4_joint04_0605_trt` | `1tdugpwj` |
| CoreFace-SGD20 e0-e7 | `coreface_sgd20_s4_0_7` | `55qv2v32` |
| CoreFace-SGD20 e8-e14 | `coreface_sgd20_s4_8_14` | `4qlyi1qt` |
| CoreFace-SubCenter-S4 e0-e19 | `coreface_subcenter_s4` | `4jsc6knz` |

新增组合的训练期公开评估 run：

| 模型 | Run 名称 | Run ID |
| --- | --- | --- |
| CoreFace-SubCenter-S4 e0-e19 | `coreface_subcenter_s4_joint_after_s2_sgd20_0605_08-06_0` | `sedie2xi` |

## Checkpoint 位置

训练输出源目录：

| 模型 | 原始 checkpoint |
| --- | --- |
| QGFace6 e9 | `/data1/dataset_0605/train_output/qgface_6_ir101_epoch11_model_realign_07-18_1/checkpoints_every_epoch/epoch:9_step:362150` |
| CoreFace-SGD20 e14 | `/data1/dataset_0605/train_output/coreface_s4_joint_after_s2_sgd20_0605_08-01_0/checkpoints_every_epoch/epoch:14` |
| SubCenter e14 | `/data1/dataset_0605/train_output/subcenter_s4_joint04_0605_07-22_1/checkpoints_every_epoch/epoch:14` |
| AdaFace e12 | `/data1/dataset_0605/train_output/s4_full_0605_06-11_0/checkpoints_every_epoch/epoch:12_step:235391` |

用于后续评估的已复制目录：

| 模型 | 目录 |
| --- | --- |
| QGFace6 e9 | `/smb_share/zkj_data/model/model_eval/qgface6_0605_9` |
| CoreFace-SGD20 e14 | `/smb_share/zkj_data/model/model_eval/coreface_sgd20_0605_14` |
| SubCenter e14 | `/smb_share/zkj_data/model/model_eval/subcenter_0605_14` |
| AdaFace e12 | `/smb_share/zkj_data/model/model_eval/adaface_0605_12` |

复制后已使用 `rsync --dry-run` 验证源目录和目标目录无差异。

新增 CoreFace-SubCenter-S4 e18 的源 checkpoint（尚未复制到上述归档目录）：

`/data1/dataset_0605/train_output/coreface_subcenter_s4_joint_after_s2_sgd20_0605_08-06_0/checkpoints_every_epoch/epoch:18`

## 共同的特征提取约定

四个 checkpoint 的 `model.pt` 都是可独立加载的 CVLFace iResNet-101 backbone：RGB `3 x 112 x 112` 输入，输出 512 维特征，dropout 为 `0.4`。四份 state dict 均有 917 项，按标准 iResNet-101 以 `strict=True` 加载成功；CPU 单张前向均输出有限的 `(1, 512)` 张量。

提取特征时不需要加载分类器分片、`pipeline.pt` 或 `qgface_loss.pt`。推理必须使用评估模式并进行 L2 归一化：

```python
model.eval()
feature = model(image)
feature = torch.nn.functional.normalize(feature, dim=1)
```

输入预处理为 RGB、`112 x 112`、`mean=[0.5, 0.5, 0.5]`、`std=[0.5, 0.5, 0.5]`。这同样适用于 AdaFace 的 `model.pt`；模型名称中的 AdaFace、QGFace、CoreFace、SubCenter 主要区分训练损失和分类器策略，不改变上述 backbone 特征接口。

## 当前联合训练配方

当前 `scripts/run_coreface_all.sh` 的新训练配方如下，供将 SubCenter 与 CoreFace 结合的实验引用：

| 阶段 | 数据增强 | 可训练部分 | 学习率 | 训练轮数 |
| --- | --- | --- | --- | ---: |
| S1 | Basic | 仅分类器 | `0.008` | 5 |
| S2 | GridSample | `body.36` 之后的 backbone，分类器冻结 | `0.008` | 20 |
| S3 | Basic | 仅分类器 | `0.006` | 5 |
| S4 | GridSample | backbone 与分类器联合训练 | backbone `0.0008`，classifier `0.00005` | 20 |

CoreFace 从 epoch 8 开始启用，因此只在 S2 和 S4 生效。S2 与 S4 使用 SGD、cosine 调度、3 epoch warmup、momentum `0.9`、weight decay `0.0005`、max grad norm `5.0`。

# 0605 模型评估与 Checkpoint 记录

记录日期：2026-08-04

本文沉淀 0605 实验中 AdaFace、QGFace6、CoreFace-SGD20、SubCenter 的横向评估结论、可用 checkpoint 与推理约定。它用于后续实验和对话的上下文，不能替代从 W&B 原始记录重新计算的结果。

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
| SubCenter | 14 | 75.966277 | 12 | 75.929335 |
| AdaFace | 12 | 75.440411 | 9 | 75.395816 |

CoreFace 的 epoch 12 综合分为 `77.081302`，与 epoch 13 和 14 非常接近。

| 模型 checkpoint | 开源 | TinyFace | IJBC | 1201 | 3T | Glint | Enhance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QGFace6 e9 | 95.833 | 72.881 | 76.007 | 74.288 | 94.281 | 68.187 | 64.838 |
| CoreFace-SGD20 e14 | 96.039 | 71.137 | 75.807 | 73.595 | 94.327 | 77.685 | 55.815 |
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

### SubCenter e14

- 综合第三；官方 IJBC 接近 CoreFace，公开集表现强，Glint 五档均值第一。
- Glint 的优势主要来自 FAR `1e-10 = 41.982`；前四级 FAR 实际由 CoreFace 领先。
- Enhance 最低，3T 五级 FAR 都最后，1201 整体偏低，跨域稳定性不及 QGFace6 与 CoreFace。

### AdaFace e12

- 综合第四，但开源集和 TinyFace 均为第一，常规公开人脸识别能力强。
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

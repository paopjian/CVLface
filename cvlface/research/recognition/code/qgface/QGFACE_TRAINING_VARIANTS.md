# QGFace 训练方案

统一入口：

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/qgface
bash scripts/examples/run_qgface_variants_0605.sh <方案编号>
```

| 编号 | Backbone | 初始化 | 分类器 | 训练策略 |
|---|---|---|---|---|
| 1 | IR-34 | 随机 | 分片全 FC | 12 epoch 联合训练 |
| 2 | IR-34 | 随机 | PFC 40% | 12 epoch 联合训练 |
| 3 | IR-101 | 随机 | PFC 40% | 12 epoch 联合训练 |
| 4 | IR-101 | AdaFace WebFace12M | PFC 40% | 12 epoch 联合微调 |
| 5 | IR-101 | AdaFace WebFace12M | PFC 40% | 分类器 5 epoch，然后冻结分类器训练 backbone |

方案 1 使用 `PartialFC_V2(sample_rate=1.0)` 作为分布式执行引擎。它不采样负类，
所有 791,509 个类别每一步都参与 softmax，因此数学上是完整 FC；区别只是类别中心按
rank 分片，每张卡保存约 1/8 权重，不会像 `FCClassifier + DDP` 那样每卡复制完整矩阵。
该配置关闭类别补齐，8 个分片的类别数总和严格为 791,509。

方案 2-5 使用 `sample_rate=0.40`：全部正类始终参与，每一步采样 40% 负类。

方案 5 第一阶段冻结 backbone，只优化随机初始化的分类器；第二阶段重新加载预训练
backbone 和第一阶段的 8 个分类器分片，冻结分类器后优化 backbone。若第一阶段已经完成，
可以直接指定 checkpoint：

```bash
STAGE1_CKPT=/data1/dataset_0605/train_output/<run>/checkpoints_every_epoch/<epoch> \
  bash scripts/examples/run_qgface_variants_0605.sh 5
```

所有方案默认使用 8 卡、每卡 batch 64、全局 identity batch 512、bf16、
`channels_last + torch.compile`。可用环境变量覆盖批量和 worker 数：

```bash
BATCH_SIZE=32 NUM_WORKERS=4 COMPILE_MODEL=false \
  bash scripts/examples/run_qgface_variants_0605.sh 3
```

QGFace 使用独立的 backbone/classifier optimizer、scheduler 和梯度裁剪预算。
随机初始化方案默认 backbone LR 为 `0.2`、classifier LR 为 `0.05`，两者各自进行
1 epoch 线性 LR warmup。可分别覆盖：

```bash
LR=0.1 CLASSIFIER_LR=0.025 \
  bash scripts/examples/run_qgface_variants_0605.sh 2
```

拆分后的 checkpoint 文件名为 `model_optimizer.pt`、`classifier_optimizer.pt` 及对应
scheduler。旧版单一 `optimizer.pt` checkpoint 不支持直接恢复优化状态，应从模型和分类器
权重启动新训练。

方案 1-3 的 backbone 和分类器均为随机初始化，不进行 classifier-only 预热；冻结随机
backbone 只会让分类器拟合无意义的随机特征。方案 4/5 使用预训练 backbone 和随机分类器，
建议优先使用方案 5 的 classifier-only 第一阶段，避免随机分类中心直接扰动预训练特征。

方案 1-4 可以先做限定 batch 的吞吐测试：

```bash
NUM_EPOCH=1 LIMIT_NUM_BATCH=50 EVALUATIONS_CONFIG=configs/skip_eval.yaml \
  bash scripts/examples/run_qgface_variants_0605.sh 1
```

方案 5 固定为 5+12 epoch，不接受 `NUM_EPOCH` 覆盖。

`IR-34/IR-101` 随机训练使用 backbone LR `0.2` 和 classifier LR `0.05`。
预训练联合训练的两者 LR 均为 `0.008`，方案 5 的 backbone 阶段使用 `0.0008`；
后两项是大规模数据微调配置，不是论文原始超参数。

## 已验证

- 方案 1 在真实 791,509 类 RecordIO 上通过 8 卡 DDP、每卡 batch 64 的前向与反向；
  feature/proxy queue 写入 1,024 条，未发生 OOM。
- 分片类别数为前 5 卡各 98,939、后 3 卡各 98,938，总和严格为 791,509。
- 方案 5 第一阶段验证 backbone 梯度为 0、分类器有梯度；第二阶段 8 个分类器分片
  全部加载成功，分类器梯度为 0、backbone 有梯度。

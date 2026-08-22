# QGFace + SubCenter 四阶段训练

## qgface5 与 qgface6 的历史区别

原 `qgface` 目录中的方案 5 使用 AdaFace WebFace12M 预训练 IR101：先冻结
backbone 训练随机 K=1 PFC 5 个 epoch，再冻结 PFC、训练 backbone 12 个 epoch。

方案 6 不是独立实验。它从方案 5 的 epoch 11 继续：先冻结 backbone 将分类器
重对齐 5 个 epoch，再冻结分类器训练 backbone 15 个 epoch。因此方案 6 依赖方案 5，
两者都不是随机初始化训练；交替冻结也会在后一阶段再次移动特征空间。

## 新实验目的

`qgface_subcenter` 不加载已有 QGFace 或 SubCenter checkpoint，只使用共同的
AdaFace WebFace12M 预训练 IR101。分类器为 K=3 SubCenter PFC，分类 logits 对每类
三个中心取最大 cosine 后应用 AdaFace margin。

四个阶段如下：

| 阶段 | Backbone | K=3 分类器 | QG 对比损失 | Epoch | 每卡 batch |
|---|---|---|---:|---:|---:|
| S1 分类器拟合 | 冻结 | 训练 | 0 | 5 | 128 |
| S2 后段训练 | 仅 `body.36` 之后训练 | 冻结 | 1 | 15 | 128 |
| S3 分类器重对齐 | 冻结 | 训练 | 0 | 5 | 128 |
| S4 全联合训练 | 全部训练 | 训练 | 1 | 15 | 64 |

S1 用于建立与预训练 backbone 对齐的 K=3 分类器。S2 和 S4 开启 QGFace 对比损失，
本脚本不运行独立对照分支。

QG 队列使用原始视图选择子中心，低质量视图共享相同的 `subcenter_id`。队列保存该
编号，历史特征做 proxy correction 时仍取同一子中心，避免低质量退化导致路由切换。

## 启动

从新目录运行：

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/qgface_subcenter
bash scripts/examples/run_qgface_subcenter_4stage_0605.sh
```

已有 S1 checkpoint 时，可用 `S1_CKPT=/path/to/stage1/checkpoint` 跳过 S1。
默认使用 8 卡、PFC 40% 负类采样。S1-S3 每卡 batch size 为 128，S4 为 64。
可通过 `NUM_GPUS`、`CUDA_DEVICES`、`S1_BATCH_SIZE`、`S2_BATCH_SIZE`、
`S3_BATCH_SIZE`、`S4_BATCH_SIZE`、`NUM_WORKERS`、`LIMIT_NUM_BATCH` 覆盖运行规模。

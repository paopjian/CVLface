# try_i200 — IR-200 glint-only 预训练调研

创建：2026-10-17（从 work_260922=adaface 项目复制改造）。wandb 项目名自动为
`try_i200`（config.py: task = 代码目录名），与主线四模型项目隔离。

## 目标

用 **iResNet-200**（此前主线 adaface 为 IR-101）在 **glint-only 数据**上从头
预训练，得到一份可复用的预训练权重，用于 IR200 规模化对照实验：

1. 后续四阶段微调的 warm-start 起点（`models.start_from=<ckpt>/model.pt`）
2. 与 TopoFR 官方 R200-Glint360K 预训练权重对比（同域同规模，损失不同）

选 glint-only 而非全量混合 train_rec 的理由（2026-10-16 调研结论）：与
TopoFR-R200 的预训练分布对齐、排除数据混杂变量；且训练量小（7.4M vs 38.8M）
先验证 IR200 从头训在本机的可行性。域适配（34t/enhance）由微调阶段补回。

## 训练集

`/data1/dataset_260918/train_rec_glint/`（2026-10-17 从全量 train_rec 提取）：

- 来源：train_v5.parquet `source_group=='glint'`，档案名 `id_NNNNNN`
- **300,194 类 / 7,396,386 图 / 22.09 GB**，JPEG 字节与原库一致
- 提取/校验脚本与记录：card8 `/root/zhaokj/extract_glint/`
  （结构流检、每类计数 vs parquet、2200 张解码抽检、100 条字节 md5 对比、
  RecordIOReader 兼容全过）

## 配方（scripts/train/pretrain_ir200_glint.sh）

| 项 | 值 | 依据 |
|---|---|---|
| 模型 | iresnet_insightface ir200（118.8M 参数，512d） | topofr 项目同款，与 TopoFR-R200 权重同构 |
| 初始化 | 随机（无 start_from） | 从头预训练 |
| 损失 | AdaFace（m=0.4, h=0.333, tα=0.01） | 主线同款 |
| 分类器 | PartialFC sample_rate=0.1 | insightface glint360k-pfc 惯例 |
| 优化器 | SGD momentum 0.9, wd 5e-4, filter_bias_and_bn | CVLface 上游从头训范式 |
| 学习率 | 0.1，warmup 2 epoch + cosine → 0 | 上游 webface4m 范式（lr 0.1 @ 有效 bs 1792） |
| batch | 224/卡 × 7 卡 = 1568 | 256 会在 GPU0 OOM：7 个 rank 各在 GPU0 建 ~384MB context（GPU0 比 GPU1-6 少 2.3GB），256 时余量 <20MB；224 腾出 ~2.6GB |
| epoch | 26 | 上游注释标准预训练长度；≈108 次身份曝光 |
| 精度 | bf16-mixed + turbojpeg 解码 | star 实测最快链路 |
| 评估 | 每 epoch 外挂 TRT fp16（eval_all_trt_single） | 与主线同口径，结果 JSON 回读 wandb |

## 评估（evaluations/configs/val_opensource_pretrain.yaml）

仅开源基准（按用户 2026-10-17 指示：参考 val_20260922.yaml，加入 IJBC、
移除全部 custom_verification4 内部项）：

| 名 | 数据 | 类型 |
|---|---|---|
| cplfw / calfw / agedb_30 | facerec_val/* | verification ACC |
| tinyface | facerec_val/tinyface_aligned_pad_0.1 | rank-1 |
| IJBC_gt_aligned | facerec_val/IJBC_gt_aligned | ijbbc 模板级 TPIR@FPIR |

注意：内部 val_34t/enhance/glint 指标不在预训练阶段监控（若需，训练完用
eval_all_trt_launcher.py 单独对 ckpt 跑 val_20260922 或 test_20260922）。

## 时间预算

锚点：ir101 bs512×7 实测 14,586 img/s（subcenter_fzcls s2 日志）。ir200
FLOPs ×1.7 → ~8.6K img/s → 7.4M 图 ≈ **15 min/epoch**；评估 ~5-8 min/epoch
（含 IJBC 模板评估与 TRT engine 首建）。全程 26 epoch ≈ **9~12 小时**。

## 产物与监控

- ckpt：`/data1/dataset_260918/train_output/ir200_glint_pretrain_MM-DD_N/checkpoints_every_epoch/epoch:N`
- 日志：`train_output/shell_logs/pretrain_ir200_glint_pretrain_<ts>.log`
- wandb：项目 `try_i200`（entity mckim），run 名 = prefix；每 epoch 记录
  4 基准 ACC/rank-1 + IJBC TPIR@FPIR 各档
- 断点：直接重跑脚本（幂等，扫最大完整 epoch 续训）；监控：
  `tail -f /data1/dataset_260918/train_output/shell_logs/pretrain_*.log`

## 微调接入（预训练完成后）

```bash
# work_260922 等微调项目里, s1 阶段把 PRETRAINED 换成本预训练产物:
PRETRAINED=/data1/dataset_260918/train_output/ir200_glint_pretrain_<...>/checkpoints_every_epoch/epoch:25/model.pt
# 注意模型包要换: models=iresnet_insightface/configs/v1_ir200.yaml (与 ir101 的
# models/iresnet 权重不互通); pefts 需带 ir200 解冻点映射 (参考 work_260922_topofr
# 的 body.72)
```

## 与本机其他训练的互斥

star 7 卡被本预训练占用期间，不要再起其他训练/评估任务（GPU7 已故障禁用）。

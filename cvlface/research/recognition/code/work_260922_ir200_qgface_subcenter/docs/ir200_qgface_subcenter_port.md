# work_260922_ir200_qgface_subcenter 移植说明

日期: 2026-10-23
基座: `work_260922_qgface_subcenter` (整目录复制, 排除 `__pycache__`)
叠加: `work_260922_ir200_subcenter` 的 IR-200 模型支持 (4 个文件)
适配机器: 8 卡机 qingsi (192.168.2.103, /root/anaconda3/envs/cvlface)

## 组合目标

把 QGFace + SubCenter (qgface_subcenter 已验证的组合) 应用到 IR-200 backbone 上,
并用**四阶段脚本中 s2 分类器冻结**的变体训练 (区别于两个源工作区 s2 分类器 0.1x lr
跟随训练的既定方案)。

## 自 work_260922_ir200_subcenter 叠加的文件

| 文件 | 作用 |
|---|---|
| `models/iresnet_insightface/__init__.py` | 注册 `ir200` → `iresnet200()` ([6,26,60,6]=98 块, 118.8M 参数) |
| `models/iresnet_insightface/configs/v1_ir200.yaml` | IR-200 模型配置 (RGB, 512d) |
| `pefts/__init__.py` | `body.K` → insightface `layerX.Y` 映射 (ir100 49 块 + ir200 98 块) + 精确前缀匹配 (`key == target or key.startswith(target+'.')`, 取代旧版 `target in key` 子串误匹配) |
| `trainers/configs/default.yaml` | 与 qgface 版仅差 3 行机器注释 (external_eval_fabric_bin 说明), 键值一致 |

`models/iresnet_insightface/model.py` 两区本就相同 (均含 iresnet200 定义), 无需动。
评估链路 `eval_all_trt_single.py` / `eval_all_trt_launcher.py` 三区完全一致, 按
checkpoint 内 config.yaml 自动发现 `iresnet_insightface` 模型, 无需改。

## 训练脚本

`scripts/train/four_stage_ir200_qgface_subcenter.sh` (CAMPAIGN=`ir200_qgface_subcenter_260922`)

### 相对 four_stage_qgface_subcenter.sh 的差异

- backbone/预训练: `models=iresnet_insightface/configs/v1_ir200.yaml` + 自产
  IR200-glint (md5=10206f92610c1305db5a2bd12914dd78)
- s2 解冻点: `body.72` (ir200 的 ~73% 深度, 对应 ir101 的 body.36)
- bs 按双视图折算对齐 ir200_subcenter 实测显存基线 (ir200 解冻段激活 ~1.7x ir101):
  - s1/s3: bs 256 (双视图=512 图/卡 ≈ ir200 s1/s3 bs512 单视图)
  - s2: bs 128 (双视图=256 图/卡 ≈ ir200 s2 bs256 单视图)
  - s4: bs 64 (双视图=128 图/卡 ≈ ir200 s4 bs128 单视图)
- 不设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (与 v7 相似度 JIT 扩展冲突,
  见 ir200 工作区 2026-10-18 排查记录)

### 本变体核心设计: s2 分类器冻结

s2 阶段 `classifiers.freeze=True` (源工作区为 False + classifier_lr=0.1x):

- `make_qgface_optimizers` 检测分类器无可训练参数 → 不创建 classifier_optimizer,
  PFC 权重 (K=3 子中心) 整个 s2 固定为 s1 训完的状态
- 分类损失梯度仍经冻结分类器回传到解冻的 backbone 段 (embeddings 有梯度、权重无)
- QGFace 对比损失的队列 proxy (`get_class_proxies`) 读冻结权重, s2 全程固定;
  子中心路由 (`route_subcenters`) 仍由原图视图特征每步实时决定
- 分类器 `train()` 不触发 (pipeline 判 `classifier.config.freeze`), PFC 的
  batch_mean/batch_std EMA 沿用 s1 末状态
- s3 再训练分类器, 消化 s2 挪动过的 backbone 特征空间 (对齐新的子中心划分)

设计动机: s2 期间 backbone 特征空间在快速移动, 分类器跟随更新会与 QGFace 的固定
proxy 队列语义相互拉扯; 冻结分类器让 s2 的对比损失锚点稳定, 把"特征对齐子中心"
留给 s3 集中解决。

### s2 断点续传说明

冻结分类器时无 classifier_optimizer → checkpoint 不写 `classifier_optimizer.pt`,
`pipelines/base.py: resume_from_dir` 对 None 模块自动跳过; classifier 本体权重
(`classifier_rank{N}.pt`) 照常保存, 阶段完成判定 (`pipeline.pt` + `model.pt` +
`classifier_rank7.pt` 齐全) 不受影响。

## 验证记录 (2026-10-23, 移植当天)

1. hydra compose: 以 s2 全量覆盖参数解析 → ir200 / TrainQGFacePipeline /
   K=3 sr0.4 / classifiers.freeze=True / quality_scale=0.2 / part_freeze body.72 ✓
2. IR-200 模型构建: 118.8M 参数 ✓
3. pefts body.72 映射: `net.layer3.40` 起解冻 (243 参数张量, 53.66% 可训练),
   `net.layer3.39` 及之前冻结, `net.features` (insightface 输出层) 解冻 ✓
4. 双优化器分支: 模型可训练+分类器冻结 → 仅 model_optimizer; 模型冻结 → 仅
   classifier_optimizer ✓
5. QGFace 组件 import: qgface loss / 双视图数据集 / identity 增强 / 队列耦合
   分类器 / train_opt.py ✓

## 事故记录 (2026-10-25): 早停广播 rank 发散 + s2 提前裁定

**现象**: s2 epoch10 评估后两个早停监视器同到 10/10 触发早停, 但 `fabric run`
退出码 1, shell 判"异常退出"自动重试 (30s 后从 epoch:10 续训)。topofr 战役
10-02 s2 epoch12 同款事故。

**根因** (两层叠加):
1. `broadcast_should_stop` 把 CUDA tensor 传给 `fabric.broadcast` → lightning
   DDPStrategy 底层走 `broadcast_object_list` (pickle 对象广播), torch 文档明确
   警告 GPU tensor 走该路径 "not well supported"。实测 rank0 收到 True、其余
   7 rank 收到 False: rank0 break 后干净退出 ('done'), 其余 rank 继续进
   `train_opt.py:756` 的 `save()`, 卡在 `pipelines/base.py:63` 的 barrier 上
   2h (NCCL watchdog 7200s) 被强杀 → rc=1。
2. 早停监视器状态不持久化: resume 后 patience/best 重置, 重试 run 会再跑若干
   epoch 才可能再次早停 (每次触发再白等 2h)。

**修复**: `broadcast_should_stop` 改广播纯 Python int (`fabric.broadcast(int(...))`,
pickle 安全路径), 2-rank CPU fabric 验证双向收到 True 干净退出。早停若再触发,
8 卡同步退出 rc=0, shell 走"正常退出但未达目标 epoch → 终止"的设计路径。
(此修复仅落本工作区; topofr/ir200_subcenter 等共用该函数的战役均已完成, 未回溯。)

**s2 提前裁定**: 早停判据被 epoch0 的异常高点锁死 (34t@1e-10 最佳 46.17 是 s2
初期 backbone 未动时的值, 之后指标先跌后回升始终未超)。自动重试 run (10-25_0)
完整跑出 epoch:11, 指标全面显著好于 epoch:10 (34t@1e-6: 68.7→79.9, @1e-10:
13.5→19.9, agedb 96.9→97.5), 人工裁定 s2 止于 epoch:11: 脚本 s2 目标改为 12
(epoch >= 11 即完成), s3 从 10-25_0/epoch:11 衔接。10-24_0 的 epoch:0~10 与
10-25_0 的 epoch:11 均原样保留可回溯。

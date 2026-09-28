# work_260922_topofr 移植说明

日期: 2026-09-28
来源: `topofr` 分支的 `code/topofr/` 工作区 (基于 work_0605 的世代; main 工作树上
该目录只剩 __pycache__ 等产物, 源码全部要用 `git show topofr:...` 读)
基座: `work_260922` (整目录复制, 排除 bench/eval_results/mlruns/wandb 等产物)
适配机器: 8 卡机 qingsi (192.168.2.103, /root/anaconda3/envs/cvlface)

## 移植内容

### 1. TopoFR 损失 (NeurIPS 2024 论文的"拓扑对齐"部分)
- `losses/topofr_loss.py`: `TopoFRLoss(margin_loss, topo_weight=0.1, use_gum=True, temp=1)`
  包装器, 作为 margin_softmax 挂在 FC / PartialFC_V2 内部 (与 adaface 同位置):
  - 基损失加 margin 后, 分类损失按 GUM 样本权重逐样本加权
  - `topo_weight>0` 且传入 input_images 时叠加持久同调拓扑损失
  - 返回 (loss, loss_dict) / AdaFace 基损失时 (loss, loss_dict, batch_mean, batch_std)
- `losses/topology.py`: 0 维持久同调 (MST persistence pairs, CPU numpy,
  O(N²)); cdist 强制 fp32 + no-mm 模式 (mm 分解有灾难性数值抵消)
- `losses/gum.py`: 1D 高斯-均匀混合 EM (CPU, 每步执行)
- `losses/__init__.py`: `get_margin_loss` 加 `topofr` 分支 (base_loss 可选
  arcface/cosface/adaface) + ArcFace/CosFace 独立类分发 (类本体在 margin_loss.py,
  work_260922 原有)
- 注意: 论文的 RFS 在线难例三元组与 manifold mixup 不在分支实现内
- 修复: margin_loss.py 的 ArcFace `self.scale`/`self.s` 属性名不一致 bug
  (原 bug 使 arcface base forward 必崩, 因 arcface base 从未被实际跑过而未暴露)

### 2. 分类器接入
- `classifiers/partial_fc/partial_fc.py`: forward 加 `input_images` 形参;
  `isinstance(margin_softmax, TopoFRLoss)` 分支走 `model_parallel=True` 的
  `DistCrossEntropyPerSampleFunc` (PartialFC label=-1 下逐样本全局 CE);
  **拓扑项只用本地 batch** (持久同调 CPU 超二次: N=128 30ms, N=1024 2289ms);
  adaface buffer 注册条件放宽到 TopoFRLoss 包装 AdaFace
- `classifiers/fc/fc.py` + 两 wrapper: 同接入 + `is_topofr` 标志暴露

### 3. Pipeline
- `pipelines/train_model_cls_topofr_pipeline.py`: 与 TrainModelClsPipeline
  差异仅 classifier 多收 `input_images=inputs`; 单优化器, 训练循环零侵入

### 4. IR200 (insightface 风格)
- `models/iresnet_insightface/model.py` 底层原有 `iresnet200` ([6,26,60,6] = 98
  blocks, 118.8M 参数); 补 `__init__.py` dispatch + `configs/v1_ir200.yaml`
- `pefts/__init__.py`: part_freeze 移植 topofr 分支的 body.K 完整映射
  (body.K 同时覆盖 legacy/insightface ir100/ir200 命名 + layerX.Y 直选;
  匹配逻辑从子串匹配改为精确/前缀匹配, 顺带修掉 body.3 误匹配 body.30 的隐患)
- `body.72` = layer1(6)+layer2(26)+layer3 前 40 块起, 与 ir101 的 body.36 同比例
  (~73% 深度); 预训练权重 `pretrained_models/recognition/topofr200/
  Glint360K_R200_TopoFR_9784.pt` (insightface 风格, 与 TreB1eN 风格
  models/iresnet 的 200 层不互通)

## 四阶段训练 (scripts/train/four_stage_topofr_ir200.sh)

骨架沿用 four_stage_260922.sh (幂等断点续传/外挂 TRT 评估/MAX_RETRY),
差异: ir200 + topofr_adaface + topofr pipeline + **bs 全程 128** + s2 解冻点
body.72; 分类器策略沿用主线 (s2/s4 classifier_lr_scale=0.1 同步训练)。

| 阶段 | epochs | bs/卡 | 训练范围 | lr | 调度 |
|---|---|---|---|---|---|
| s1 | 5 | 128 | PFC 随机初始化拟合 (backbone 冻结) | 0.008 | step [2,4] λ0.3 |
| s2 | 15 | 128 | body.72+ + 分类器 0.1x | 0.008 | cosine warmup2 |
| s3 | 5 | 128 | PFC 重训 (backbone 冻结) | 0.006 | step [2,4] λ0.3 |
| s4 | 15 | 128 | 全模型 + 分类器 0.1x | 0.0008 | cosine warmup2 |

与 topofr 分支原配方 (train_topofr_ir200_0605.sh) 的差异: 分支 s2 冻结分类器、
s4 原 config 冻结分类器 (joint 变体 classifier_lr=0.0001); 本脚本按 260922 主线
惯例 s2/s4 分类器 0.1x lr 同步训练。

关键约束:
- **bs 勿随意调大**: 拓扑损失 O(N²) 本地 batch CPU 计算, bs 翻倍拓扑耗时翻两番;
  GUM EM 也在 CPU。两者合计约 +15~20% 步时间 (分支文档口径)
- 拓扑项 latent_norm 在 topology.py 硬编码 `.cuda()`, 训练必须见 GPU
- ir200 显存 ~1.7x ir101; OOM 时优先降 bs (同时降拓扑耗时)

## 验证记录 (2026-09-28)

- `test_topofr.py` (分支集成测试移植 + 2 个新增测试, 6 项全过): 导入/损失创建/
  配置加载/前向传播/PartialFC+TopoFR 分布式路径 (is_topofr 标志、adaface buffer、
  model_parallel 逐样本 CE、拓扑项反向)/ir200 构造+body.72 冻结范围
- 单卡 GPU 端到端 2-batch 训练 (synthetic + topofr 全配置 + ir200): 见下节结果

## 已知事项

- work_260922 里未提交的 eval_all_trt_single.py / evaluations/__init__.py 改动
  已随复制带入 (属当前在用状态)
- GUM 的 entropy 成对取负逻辑 (entropy_np[2*arange(N//2)] *= -1) 是分支成对增强
  训练的遗留, 标准训练沿用 (N 为奇数时最后一个样本不参与符号翻转)
- 拓扑损失对显存不敏感 (CPU 计算), 但 GUM EM 与持久同调的 CPU 开销在高 bs 下
  会放大; 若 CPU 成为瓶颈可考虑 topo_weight=0 关闭拓扑项做对照

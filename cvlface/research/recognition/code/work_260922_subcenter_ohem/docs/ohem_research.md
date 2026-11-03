# OHEM (Online Hard Example Mining) 接入研究与实现

工作区: `work_260922_subcenter_ohem` (由 `work_260922_subcenter` 复制, 2026-10-28)
结论先行: **可以接入, 且接入点非常干净**。本区已在 `DistCrossEntropy` 中实现分类版
OHEM (per-sample top-k 困难样本聚合), 默认 `ohem_ratio=0.7` + 1000 步线性退火,
四阶段脚本 `scripts/train/four_stage_subcenter_ohem.sh` 一键启动, 冒烟测试 7 项全过
(`test_ohem_smoke.py`)。

---

## 1. 链路分析: 为什么接入点在 DistCrossEntropy

训练链路 (`train_opt.py` → `TrainModelClsPipeline.__call__`):

```
model(inputs) → feats
classifier(feats, targets) → PartialFC_V2.forward:
    all_gather embeddings/labels (跨卡拼成全局 batch)
    → sample() 负类中心采样 (PFC sr=0.4, 正类中心必选)
    → compute_logits (cosine, SubCenter K=3 子中心 amax)
    → margin loss (AdaFace: 按特征范数自适应加 margin, 只改 logits)
    → DistCrossEntropy(logits, labels)   ← OHEM 接在这里
```

`DistCrossEntropyFunc.forward` (改造前) 的关键结构:

```python
loss = torch.zeros(batch_size, 1)          # 每样本"真类概率"向量 (B,1)
loss[index] = logits[index].gather(1, label[index])
distributed.all_reduce(loss, SUM)           # 各 rank 位相同的完整每样本 loss
return loss.clamp_min_(1e-30).log_().mean() * (-1)   # ← .mean() 之前, 每样本量已就绪
```

两点使得 OHEM 几乎零成本接入:

1. **每样本 loss 向量已经存在**, 只是随即被 `.mean()` 聚合掉。OHEM 要做的只是把
   "全批均值" 换成 "最难的 k 个样本的均值"。
2. **该向量经 all_reduce 后各 rank 位相同** (NCCL allreduce 保证跨 rank 位一致),
   top-k 选择在各卡上独立计算结果也一致, 不需要任何额外通信。

margin loss (AdaFace/ArcFace) 在此之前已经生效, 所以"难度"是在**加 margin 之后**
的 loss 上度量的 —— 与 OHEM 文献中的用法一致 (在最终分类 loss 上做挖掘)。

## 2. 实现方案

改动集中在 `classifiers/partial_fc/partial_fc.py`:

| 位置 | 改动 |
|---|---|
| `DistCrossEntropyFunc.forward` | 计算 `log_prob` 后, 若 `ratio_eff<1` 则 `topk(log_prob, k, largest=False)` 选最难样本, `out_loss = mean(log_prob[kept]) * (-1)`; 同时产出统计张量 |
| `DistCrossEntropyFunc.backward` | 手工梯度从 `÷batch_size` 改为行级权重: 选中行 `×1/k`, 未选中行 `×0` (见下方推导) |
| `DistCrossEntropy` (module) | 持有 `ohem_ratio / ohem_min_kept / ohem_warmup_steps`; `ohem_step` 计数 buffer 随 classifier checkpoint 存取, 断点/跨阶段不丢退火进度 |
| `PartialFC_V2.__init__` | 新增三个 OHEM 形参 (默认关闭) |
| `classifiers/partial_fc/__init__.py` | `from_config` 用 `getattr(classifier_cfg, 'ohem_ratio', 1.0)` 等透传, **旧 yaml 不写这些键 = 完全旧行为** |
| `train.py` / `train_opt.py` | `log_classifier` 新增 5 个 OHEM 指标 |

### 2.1 backward 的梯度对齐 (核心正确性)

原 backward 返回 `(softmax - onehot) / B`, 对应 forward 的 `mean over B`。OHEM 下
forward 是 `mean over kept`, 所以 backward 改为行级权重 `w_i` (选中 `1/k`, 未选中 `0`)
乘到 `(softmax - onehot)` 上。注意两行都要乘:

- **本 shard 拥有 label 的行**: 梯度 `(p - onehot)/k` (选中) 或 `0` (未选中);
- **label 在别的 shard 的行**: 原实现里这些行也有 `p/B` 梯度 —— 这是 softmax
  **分母** 对 loss 的贡献 (每卡的 logits 都进了全局归一化项)。OHEM 下未选中样本
  对 loss 无贡献, 分母梯度同样为 0; 选中样本分母梯度为 `p/k`。

冒烟测试 [2] 用纯 autograd 参考实现 (softmax→per-sample NLL→topk 均值) 对拍梯度,
max diff < 1e-5, 且未选中行梯度严格为 0。

### 2.2 warmup 与断点续训

- `ohem_warmup_steps>0` 时, 有效 ratio 从 1.0 (标准 CE, 与基线零差异) 线性退火到
  `ohem_ratio`。s1 分类器随机初始化, 冷启动阶段大多数样本都"难", 直接收紧会放大
  噪声, 退火是保险。
- `ohem_step` 是 buffer, 随 `classifier_rank*.pt` 一起保存; 阶段内断点续训
  (`trainers.resume`) 与阶段间衔接 (`pefts.classifier_ckpt_dir`) 都会带着走
  (两条加载路径均为 `strict=False`, 新旧 checkpoint 互兼容)。副作用: warmup 只在
  s1 发生一次, s2/s3/s4 直接以目标 ratio 起步 —— 各阶段起始都是已收敛模型, 这是
  合理的; 若想让每阶段重新退火, 在阶段脚本里把 `ohem_warmup_steps` 覆盖为该阶段
  步数即可。

### 2.3 精度路径 (bf16-mixed)

`ohem_weight` 在 forward 里 cast 成 `logits.dtype`, backward 的 in-place `mul_`
不会触发 bf16×fp32 类型提升错误 (冒烟测试 [5]); OHEM 分支的 loss 聚合走 fp32
(`log_prob.float()`), 数值上不劣于原实现 (原实现 bf16 下 mean)。

## 3. 配置与使用

新配置 `classifiers/configs/partial_fc_subcenter_k3_sample40_ohem.yaml`
(SubCenter K=3 + sr=0.4 不变, 追加 OHEM 三键):

```yaml
ohem_ratio: 0.7          # 每步保留最难的 70% 样本; 1.0=关闭; 建议 0.7~0.9
ohem_min_kept: 256       # 保留数下限 (全局 batch 3584 时基本不触发, 保护小 batch)
ohem_warmup_steps: 1000  # 从 1.0 线性退火到 0.7 的步数; 0=立即生效
```

训练 (与 subcenter 基线四阶段完全同构, 只换 classifier 配置和 campaign 名):

```bash
bash scripts/train/four_stage_subcenter_ohem.sh
```

临时调参无需改 yaml, hydra 直接覆盖 (脚本头有说明):

```bash
classifiers.ohem_ratio=0.8 classifiers.ohem_warmup_steps=2000
```

新增日志 (wandb/csv, `log_classifier` 每 200 batch):

| 键 | 含义 |
|---|---|
| `train/ohem_loss_all` | 同批**全样本** NLL 均值 —— 跨 run 对比用这个 |
| `train/ohem_loss_kept` | 选中困难样本 NLL 均值 (≈`train/loss`) |
| `train/ohem_kept` / `train/ohem_batch` | 本步保留数 / 全局 batch |
| `train/ohem_ratio_eff` | 退火中的有效 ratio |

**注意: 开 OHEM 后 `train/loss` 只聚合困难样本, 数值天然高于 subcenter 基线,
不能直接横向对比**, 看 `train/ohem_loss_all` 与各验证集指标。

## 4. 与现有技术的交互 (研究结论)

- **AdaFace (本区在用)**: AdaFace 本身就是"难度自适应"—— 按特征范数给难样本更大
  margin (低质量图范数小 → margin 大)。OHEM 与它是**正交叠加**: AdaFace 调整
  每样本的 loss *形状* (margin), OHEM 调整 loss 的*聚合权重* (哪些样本回传)。
  叠加有效但会双重偏向困难样本, 所以 ratio 从温和的 0.7 起步, 不建议低于 0.5。
- **SubCenter K=3**: logit 取子中心 amax, 每样本难度度量的口径随之改变 (与最像
  的子中心比), 与 OHEM 无冲突。
- **PFC sr=0.4**: 每步难度是在"当次采样的负类子集"上估计的, 比全类 softmax 噪声
  略大 (某步恰好采到强负类的样本会被高估难度)。top-k 在 3584 的全局 batch 上
  平均掉了这个噪声, 实践可忽略。
- **interclass_filtering_threshold (losses yaml)**: 那是"脏 logit 过滤"(处理脏负类),
  与 OHEM 的样本级挖掘是两个维度, 目前 adaface.yaml 里为 0 未启用, 无冲突。
- **风险 — 标签噪声**: 大规模人脸数据 (WebFace12M 系) 的困难样本里混有错标/低质
  图, ratio 压得过低会拟合噪声, 典型症状是训练 loss 降而验证集(尤其 tinyface/
  高难集以外)不升。这也是建议 0.7 而非检测领域常用 0.3 的原因 (检测的"难样本"
  多为有价值的尺度/遮挡变体, 分类的"难样本"更可能是噪声)。
- **梯度尺度**: mean-over-selected 使每个被选样本的梯度 ×(B/k) (~1.43x @0.7),
  等效于对困难样本小幅加学习率; 有 warmup 缓冲, 一般无需动 lr。若观察到震荡,
  先降 ratio 再考虑 `optims.lr` 减 10~20%。

## 5. 验证记录

`test_ohem_smoke.py` (单进程 gloo, 不依赖数据集, `python test_ohem_smoke.py` 可重跑):

1. `ohem_ratio=1.0`: loss/梯度与 `F.cross_entropy` 一致 —— **不开启时与原实现零差异**
2. `ratio=0.5`: loss == top-k NLL 均值; 梯度与 autograd 参考一致; 未选中行梯度为 0
   (开发中该测试抓到过一个真实 bug: log_prob 取 topk 最大值选中了最容易样本, 已修)
3. warmup: ratio_eff 1.0 → 目标值线性退火并保持
4. 断点: `ohem_step` 随 state_dict 存取; 无该键的旧 checkpoint 兼容
5. bf16: 无 dtype 报错, loss 与 fp32 参考一致
6. stats 四元组语义正确
7. `ohem_min_kept` 下限生效

## 6. 实验策略 (2026-10-28 领导指示定稿)

背景观察 (领导看错题集): FA 多为**鼻子/嘴巴有肉眼可见差异**的对比对 —— 人眼容易区分、
模型却给出高相似度, 说明 embedding 的局部细节 (鼻嘴差异) 没学到。归因: 这类
lookalike 对在训练时各自都是"容易样本" (类内收敛良好), 互相之间的相似性没有产生
足够的 loss 信号, 梯度从未专门把它们推开。

领导的判断与 OHEM 机制的对应:

- "更多是同 batch 里的难例, 所以要做 OHEM" —— 在本链路 (分类式 PFC 训练) 里,
  "同 batch" 的实质是**当步采样进 logit 的负类中心集合**: 某样本的 lookalike 类
  中心恰好被采样 (sr=0.4, 每步 40% 概率) 且相似度高时, 该样本的分母里出现强竞争
  者 → per-sample loss 变大 → OHEM 恰好把这些步的样本保留并放大梯度, 正对准
  lookalike 推开这一学习信号。
- "只是把难例加进训练集, 不一定分到同 batch, 可能学不到" —— 完全正确, 而且在
  本链路里还有一个更强的版本: 即使 lookalike 图进了 batch, 它的**类中心** 若不在
  当步的 40% 采样集里, 这一步对这对 lookalike 的互相排斥仍然学不到。离线加数据
  无法解决这个"当步交互"问题, 在线挖掘 (OHEM) 是对的方向。
  - 可选后续增强 (本轮不动, 避免双变量): 提高 `classifiers.sample_rate`
    (0.4 → 0.5/0.6), 直接提高 lookalike 中心参与概率; 代价是 K=3 下 logit
    显存/耗时线性上涨, s4 可能再触 OOM, 需配合 bs 降档。

训练策略 (已落进 `four_stage_subcenter_ohem.sh`):

- **加 OHEM + 拉长轮次, 只要不明显掉点就训到完全没收益**:
  - s2/s4 上限 15 → 30 epoch, cosine 调度同步拉长;
  - 早停即完成: 验证指标连续 10 epoch 无改进 (train_opt.py 内建 patience=10,
    监控 summary mean + work_260922_34t/tpir_at_far_1e-10) 时早停收工;
    已练 >= 15 epoch (不低于 subcenter 基线训练量) 才接受为完成, 否则中止待查;
    早停完成打 `earlystop.done` 标记, 整脚本重跑不误续训 (幂等)。
  - s1/s3 不拉长: 分类器-only 阶段 backbone 冻结, 验证用的 embedding 不变,
    指标无信号、早停不会触发, 拉长轮次无法证伪, 维持 5 epoch。

观察与判定:

- 主对照: subcenter 基线 (无 OHEM, s2/s4 各 15ep) vs 本战役 (OHEM 0.7 + 拉长)。
- 看 `train/ohem_loss_all` 下降斜率是否快于基线 `train/loss`;
- 重点看 FA 集趋势: 34t/enhance/glint 的 tpir@1e-10、以及错题集里鼻嘴型 FA 对
  的相似度是否被压低 (可用 portable_eval 对固定 FA 对清单批量复测);
- 消融 (按优先级): ratio 0.9/0.8/0.7; 只 s2/s4 开 OHEM; warmup 0 vs 1k;
  sample_rate 0.5 (与 OHEM 解耦单独跑)。

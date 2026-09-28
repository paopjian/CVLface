# work_260922_qgface_subcenter 移植说明

日期: 2026-09-24
来源: `qgface_subcenter` 分支的 `code/qgface_subcenter/` 工作区 (基于 work_0605 的旧世代)
基座: `work_260922` (整目录复制, 排除 bench/eval_results/mlruns/wandb 等产物)
适配机器: 8 卡机 qingsi (192.168.2.103, /root/anaconda3/envs/cvlface)

## 移植的三项技术

1. **QGFace 质量引导对比损失** (`losses/qgface.py`, 分支原样)
   - 双视图 (query=降质 / key=原图) 对比学习, 负样本来自 proxy 实时队列
   - 队列 proxy correction: 分类器中心移动后按范数比例平移历史特征
   - 质量权重来自 AdaFace margin_scaler (EMA norm 统计), `quality_scale_method=sgn`
     低质量对保留全梯度、高质量对关闭对比项
   - `losses/adaface.py` 增加 `quality_scale` (≠0 时 sigmoid 质量门控缩放分类 logit),
     `losses/configs/qgface_adaface.yaml` 设 0.2

2. **SubCenter PFC 分类器** (K=3)
   - `classifiers/partial_fc/partial_fc.py`: `num_subcenters`/`pad_classes` 参数,
     weight 形状 `(num_local*K, D)`, `sample()` 子中心索引展开, `compute_logits()`
     对子中心 amax 后再进 margin_softmax
   - `classifiers/partial_fc/__init__.py`: 分支的三个队列耦合接口
     `route_subcenters` (原图视图特征选子中心) / `get_class_proxies` (取子中心行
     作 proxy) / `get_margin_scaler` (复用 PFC 上的 AdaFace norm 统计)
   - `classifiers/base/__init__.py`: K 感知 checkpoint 加载 (形状校验 + 跨卡数
     重分配按 `class_start*K` 切行)
   - `classifiers/fc/__init__.py`: 同接口的 K=1 退化实现
   - 配置: `classifiers/configs/partial_fc_subcenter_k3_sample40.yaml` (sr=0.4, K=3)

3. **双视图训练 pipeline** (`pipelines/train_qgface_pipeline.py`)
   - 两视图拼接过同一 backbone, 分类损失双视图都参与 (K>1 时 max-subcenter)
   - `contrast_weight>0` 时: 全局 gather → 子中心路由 (两视图共享 id) → 入队 →
     proxy correction 出队 → QGFaceLoss (D2N 双向配对)
   - backbone 与 PFC 独立优化器/裁剪预算 (`optims/optims.py: make_qgface_optimizers`,
     分类器基准 lr 走 `optims.classifier_lr`, 裁剪走 `optims.classifier_max_grad_norm`)
   - 数据链路: `dataset/contrastive_view_dataset.py` + `dataset/qgface_view_transform.py`
     (0.1-0.5 下采样 + JPEG75 + crop/photometric), 挂钩在 `dataset/__init__.py`
     `get_train_dataset` 尾部 (`data_augs.contrastive_views: true` 触发)
   - 底层增强 `identity_v2_numpy` (key 视图保持干净), 配置 `data_augs/configs/qgface.yaml`

## train_opt.py 的条件分支 (非 QGFace 路径零改动)

`is_qgface_pipeline = cfg.pipelines.name == 'TrainQGFacePipeline'` 时:
双优化器构造 / 双调度器 (make_scheduler 新增 base_lr/warmup_steps 可选参) /
fabric.setup(model, model_optimizer) + setup_optimizers(classifier_optimizer) /
pipeline_from_config 传 kwargs+fabric / 训练循环 set_epoch + 双 step 双 clip 双
scheduler / log_dict 合入 qgface/* 分项指标。分类器单独训练阶段 (s1/s3) 不建
dummy_model, backward sync 走 nullcontext。

## 四阶段训练 (scripts/train/four_stage_qgface_subcenter.sh)

骨架沿用 four_stage_260922.sh (幂等断点续传/外挂 TRT 评估/MAX_RETRY), 差异:
- 全阶段 QGFace 三件套配置 (双视图 augs + qgface_adaface + train_qgface pipeline + K=3 分类器)
- bs 相对基线减半 (双视图下每卡过 backbone 图像数与基线持平): s1-s3 256, s4 128
- 阶段表 (分类器策略沿用主线 0.1x 同步训练; 分类器 lr 走 classifier_lr 绝对值):

| 阶段 | epochs | bs/卡 | contrast_weight | 训练范围 | lr (model/classifier) | 调度 |
|---|---|---|---|---|---|---|
| s1 | 5 | 256 | 0.0 | 分类器 K=3 随机初始化 | 0.008 / 0.008 | step [2,4] λ0.3 |
| s2 | 15 | 256 | 1.0 | body.36+ + 分类器 0.1x | 0.008 / 0.0008 | cosine warmup2 |
| s3 | 5 | 256 | 0.0 | 分类器 | 0.006 / 0.006 | step [2,4] λ0.3 |
| s4 | 15 | 128 | 1.0 | 全模型 + 分类器 0.1x | 0.0008 / 0.00008 | cosine warmup2 |

- 与分支原配方 (run_qgface_subcenter_4stage_0605.sh) 的差异: 分支 s2 冻结分类器
  (之后 s3 重对齐), 本脚本按主线惯例 s2/s4 分类器 0.1x 同步训练
- 约束: queue_size 8192 >= 2x 全局 batch (s1-s3 为 4096); K=3 checkpoint 勿与
  K=1 campaign 混用

## 验证记录 (2026-09-24)

- `test_qgface.py` (分支移植, 7 项 CPU 测试): loss/队列、双优化器、pipeline 前向、
  classifier-only、断点存取、真 PartialFC K=3 路由/proxy、真 ir18+fabric 全链路 — 全过
- 单卡 CPU 端到端 2-batch 训练 (synthetic + qgface 全配置): 训练/保存正常,
  checkpoint 含 pipeline.pt + model.pt + classifier_rank0.pt + qgface_loss.pt +
  model/classifier_optimizer.pt + 双 scheduler.pt

## 已知事项

- work_260922 里未提交的 eval_all_trt_single.py / evaluations/__init__.py 改动
  已随复制带入 (属当前在用状态)
- eval 输出目录为 `eval_results` (本机原样; star 机版本改过 eval3_results)
- s1/s3 的 identity 底层增强与基线 basic/gridsample 不同: QGFace 设计上 key 视图
  需保持干净以提供稳定质量信号

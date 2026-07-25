# Sub-center AdaFace 训练说明

## 原理与实现对应

InsightFace 的 Sub-center ArcFace 为每个身份维护 `K` 个单位球面原型。对样本 `x` 和身份
`c`，分类 logit 为：

```text
cos(theta_c) = max_k cosine(x, W[c, k])
```

随后才对目标身份的聚合 logit 应用 margin。原始实现推荐 `K=3`，这样主子中心学习身份的
干净分布，其余子中心可吸收姿态、域差异及标签噪声。这里保留同样的 max 聚合，只把固定
ArcFace margin 替换为 AdaFace 的质量自适应 margin。

PartialFC 的采样单位仍是身份类：采中一个身份时会同时取出其全部 `K` 个子中心。因而
`sample_rate=0.4` 表示采样 40% 的身份，而不是 40% 的中心。分类器 checkpoint 中每个身份
的 `K` 行连续存放，支持在 GPU 数改变时按身份边界重新分片。

## dataset_0605 训练

训练所需数据和评估项沿用 `流程.txt`：RecordIO 数据
`/data1/dataset_0605/train_rec`、791509 类，训练输出到
`/data1/dataset_0605/train_output`，每轮评估使用
`evaluations/configs/val_20260605.yaml`。

正式训练使用五阶段启动脚本：

```bash
cd "/root/zhaokj/CVLface/cvlface/research/recognition/code/try_subcenter"
bash "scripts/examples/run_subcenter_5stage_0605.sh"
```

脚本依次执行分类器 0.4 预热、`body.36` 微调、分类器 0.4 重对齐、全模型与分类器联合训练，
最后冻结 backbone 并用 `sample_rate=1.0` 对齐清洗分类器。每个阶段自动选择上一阶段最大
epoch，详细参数和断点启动方式见 `流程.txt`。

## 已验证的轻量 smoke

以下命令已在 2 张 RTX 4090 上完成 1 个 batch，用 1000 类合成数据验证 NCCL、AdaFace
前后向、优化器和 checkpoint，不会生成 791509 类的大分类器：

```bash
env CUDA_VISIBLE_DEVICES=0,1 DATA_ROOT=/tmp \
  LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib \
  DECODE_BACKEND=turbojpeg \
  /root/anaconda3/envs/cvlface/bin/fabric run \
  --devices=2 --precision=bf16-mixed train_opt.py \
  trainers.prefix=subcenter_train_opt_nostep_smoke \
  trainers.num_gpu=2 trainers.batch_size=16 trainers.num_workers=0 \
  trainers.precision=bf16-mixed trainers.limit_num_batch=1 \
  trainers.skip_final_eval=True trainers.using_wandb=False \
  models=iresnet/configs/v1_ir18.yaml models.freeze=True \
  dataset=configs/synthetic.yaml dataset.num_classes=1000 dataset.num_image=128 \
  +dataset.model_save_dir=/tmp/cvlface_subcenter_nostep_smoke \
  data_augs=configs/basic_v2_numpy.yaml \
  classifiers=configs/partial_fc_subcenter_k3_sample40.yaml \
  losses=configs/adaface.yaml pefts=configs/freeze.yaml \
  optims=configs/step_sgd.yaml optims.lr=0.01 optims.num_epoch=1 \
  evaluations=configs/skip_eval.yaml
```

验证结果：loss 为 `35.00`；checkpoint 成功保存为 `checkpoints_every_epoch/epoch:0/`，两个
rank 的分类器权重均为 `[1500, 512]`，即每卡 `500` 个身份、每身份 `3` 个子中心。

## 与原论文数据清洗步骤的区别

InsightFace 完整方案会在 K=3 训练后统计每类的主导子中心，丢弃与主导中心夹角大于 75 度
的样本，再用 K=1 重训。当前实现完成的是 Sub-center AdaFace 训练阶段，没有自动改写
37M 图像的 RecordIO。清洗属于不可逆的大规模数据操作，应在单独输出目录中实现和验证，
不能覆盖 `/data1/dataset_0605/train_rec`。

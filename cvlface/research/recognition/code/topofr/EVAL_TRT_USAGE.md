# TRT 评估脚本使用说明

## 概述

`eval_all_trt_launcher.py` 和 `eval_all_trt_single.py` 是用于对训练好的 checkpoint 进行 TensorRT 加速评估的脚本。

## 修改内容

1. **模型加载修复**: 移除了 `get_model()` 调用中的 `'work_0605'` 参数，因为 topofr 使用自己的 iresnet_insightface 模型实现
2. **适配 checkpoint 结构**: 脚本现在可以正确加载 topofr 训练输出的 checkpoint

## 使用方法

### 评估 s2_topofr_body36_0605_09-03_0

```bash
python eval_all_trt_launcher.py \
  --num_gpu 8 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s2_topofr_body36_0605_09-03_0
```

### 评估 s4_topofr_full_0605_09-04_0

```bash
python eval_all_trt_launcher.py \
  --num_gpu 8 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s4_topofr_full_0605_09-04_0
```

## 参数说明

- `--num_gpu`: 使用的 GPU 数量（默认: 7）
- `--eval_config_name`: 评估配置文件名（默认: test_20260605）
  - 配置文件位于 `evaluations/configs/test_20260605.yaml`
  - 包含多个测试集: work_0605_3t, work_0605_enhance, work_0605_glint, work_1201, IJBC_gt_aligned, ijbc
- `--ckpt_dir`: checkpoint 目录路径（必需）
  - 脚本会自动遍历该目录下所有 epoch checkpoint
- `--project_name`: wandb 项目名称（默认: work_0605_test）
- `--name`: 本次评估的运行名称（必需）
  - 建议使用与 checkpoint 目录相同的名称
- `--precision`: TRT engine 精度（默认: fp16）
  - `fp16`: 更快，但有数值精度损失
  - `fp32`: 更慢，但更接近 PyTorch 原始结果
- `--timeout_minutes`: 单个 checkpoint 评估超时时间（默认: 90分钟）
- `--max_retries`: 失败后重试次数（默认: 2）

## Checkpoint 结构要求

每个 checkpoint 目录需要包含：
- `model.yaml`: 模型配置文件
- `model.pt`: 模型权重文件

示例结构：
```
checkpoints_every_epoch/
├── epoch:0/
│   ├── model.yaml
│   ├── model.pt
│   ├── config.yaml
│   └── ...
├── epoch:1/
│   └── ...
```

## 支持的模型类型

topofr 使用的是 `iresnet_insightface` 模型：
- IR-18
- IR-50
- IR-101

模型配置示例（从 checkpoint 中的 model.yaml）：
```yaml
name: ir101
output_dim: 512
yaml_path: /iresnet_insightface/configs/v1_ir101.yaml
```

## 评估流程

1. **构建 TRT engine**: 主进程在 GPU 0 上将 PyTorch 模型转换为 TensorRT engine
2. **多 GPU 特征提取**: 每个 GPU 加载 TRT engine，提取各自分片的特征
3. **特征聚合**: 主进程从 `/dev/shm` 收集所有特征
4. **指标计算**: 根据不同的 evaluation_type 计算相应指标
5. **结果保存**: 保存 CSV 文件并上传到 wandb

## 输出位置

结果保存在：
```
eval3_results/<name>/
├── epoch_0_raw.csv
├── epoch_0_summary.csv
├── epoch_1_raw.csv
├── epoch_1_summary.csv
└── ...
```

同时会自动上传到 wandb 项目中。

## 断点续评

脚本支持断点续评功能：
- 如果 wandb 中已存在同名 run，会自动读取已完成的 epoch
- 只评估未完成的 checkpoint
- 结果继续追加到同一个 wandb run

## 测试模型加载

在运行完整评估前，可以先测试模型是否能正常加载：

```bash
python test_model_load.py
```

这会测试两个 checkpoint 的模型加载和前向传播。

# TopoFR TRT 评估脚本修改总结

## 修改日期
2026-09-06

## 修改内容

### 1. 核心修改：`eval_all_trt_single.py`

**位置**: 第 679 行

**修改前**:
```python
model = get_model(model_config, 'work_0605')
```

**修改后**:
```python
model = get_model(model_config)
```

**原因**: 
- topofr 使用自己的 `iresnet_insightface` 模型实现，不是 work_0605 的通用模型
- `get_model()` 函数的 `task` 参数实际上未被使用
- checkpoint 中的 `model.yaml` 指定了正确的 `yaml_path: /iresnet_insightface/configs/v1_ir101.yaml`

### 2. 适配的 Checkpoint

脚本现在可以正确评估以下 checkpoint：

1. **s2_topofr_body36_0605_09-03_0**
   - 路径: `/data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch`
   - 模型: IR-101 (iresnet_insightface)
   - 从 warmup checkpoint 微调而来

2. **s4_topofr_full_0605_09-04_0**
   - 路径: `/data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch`
   - 模型: IR-101 (iresnet_insightface)

### 3. 创建的辅助文件

1. **test_model_load.py** - 测试模型加载是否正常
2. **test_eval_trt.sh** - 快速验证单个 epoch 评估
3. **verify_modifications.py** - 验证所有修改的完整性
4. **run_eval_examples.sh** - 完整的评估命令示例
5. **EVAL_TRT_USAGE.md** - 详细使用文档

## 使用流程

### 步骤 1: 验证修改
```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr
python verify_modifications.py
```

### 步骤 2: 测试模型加载（可选）
```bash
python test_model_load.py
```

### 步骤 3: 运行完整评估

**评估 s2_topofr_body36_0605_09-03_0**:
```bash
python eval_all_trt_launcher.py \
  --num_gpu 8 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data1/dataset_0605/train_output/s2_topofr_body36_0605_09-03_0/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s2_topofr_body36_0605_09-03_0
```

**评估 s4_topofr_full_0605_09-04_0**:
```bash
python eval_all_trt_launcher.py \
  --num_gpu 8 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s4_topofr_full_0605_09-04_0
```

## 技术细节

### Checkpoint 结构
```
checkpoints_every_epoch/
├── epoch:0/
│   ├── model.yaml          # 模型配置（包含 yaml_path）
│   ├── model.pt            # 模型权重
│   ├── config.yaml         # 训练配置
│   ├── optimizer.pt        # 优化器状态
│   ├── lr_scheduler.pt     # 学习率调度器
│   ├── classifier_rank*.pt # 分类器权重（多卡）
│   └── pipeline.pt         # Pipeline 状态
```

### 模型配置示例 (model.yaml)
```yaml
input_size: [3, 112, 112]
color_space: RGB
name: ir101
output_dim: 512
start_from: /path/to/warmup/checkpoint/model.pt
freeze: true
yaml_path: /iresnet_insightface/configs/v1_ir101.yaml
```

### 评估配置 (test_20260605.yaml)
包含 6 个测试集：
1. work_0605_3t - custom_verification4
2. work_0605_enhance - custom_verification4
3. work_0605_glint - custom_verification4
4. work_1201 - custom_verification4
5. IJBC_gt_aligned - ijbbc (官方协议)
6. ijbc - ijbc_custom (自定义协议)

### TRT 加速流程
1. **主进程**: PyTorch → ONNX → TensorRT engine（一次性）
2. **多进程**: 每个 GPU 加载 engine，并行提取特征
3. **聚合**: 通过 `/dev/shm` 共享内存收集特征
4. **计算**: 各测试集的指标计算
5. **保存**: CSV 文件 + wandb 上传

### 精度选项
- `--precision fp16`: 更快，适合大规模评估（默认）
- `--precision fp32`: 更稳定，更接近 PyTorch 原始结果

## 注意事项

1. **环境依赖**: 需要 TensorRT 库，脚本会自动设置 `LD_LIBRARY_PATH`
2. **显存需求**: 每个 GPU 约需 3-4GB 显存（batch_size=256）
3. **断点续评**: 支持自动跳过已评估的 epoch
4. **超时设置**: 默认 90 分钟/checkpoint，可通过 `--timeout_minutes` 调整
5. **重试机制**: 失败后自动重试 2 次，可通过 `--max_retries` 调整

## 验证清单

- [x] 移除 `get_model()` 中的 `'work_0605'` 参数
- [x] 验证 checkpoint 路径存在
- [x] 验证模型配置文件格式
- [x] 验证评估配置文件存在
- [x] 创建测试脚本
- [x] 创建使用文档
- [x] 创建验证脚本

## 预期结果

每个 checkpoint 评估完成后会生成：
- `eval3_results/<name>/epoch_X_raw.csv` - 原始结果
- `eval3_results/<name>/epoch_X_summary.csv` - 汇总结果
- wandb 面板中的实时图表

评估指标包括：
- TPIR @ FAR (1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
- IJB-C TAR @ FAR
- Verification accuracy

## 故障排除

如果遇到问题：

1. **模型加载失败**: 运行 `python test_model_load.py` 检查
2. **TRT 构建失败**: 检查 CUDA/TensorRT 版本兼容性
3. **OOM 错误**: 减少 `BATCH_SIZE`（默认 256）或 GPU 数量
4. **评估卡住**: 检查 `/dev/shm` 空间是否充足
5. **wandb 上传失败**: 检查 `WANDB_TOKEN` 环境变量

## 相关文件

- `eval_all_trt_single.py` - 单个 checkpoint 评估主脚本
- `eval_all_trt_launcher.py` - 批量评估启动器
- `evaluations/configs/test_20260605.yaml` - 评估配置
- `models/iresnet_insightface/__init__.py` - 模型定义

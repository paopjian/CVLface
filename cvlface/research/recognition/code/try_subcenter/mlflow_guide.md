# MLflow 离线实验追踪指南

## 概述

使用 MLflow + SQLite 本地离线记录训练实验数据，无需常驻服务，无泄密风险。

## 环境

```bash
# 已安装在 cvlface conda 环境中
conda activate cvlface
pip install mlflow  # 如需重装
```

## 集成位置

`train5.py` 中已集成 MLflow，训练时自动记录以下内容：

| 阶段 | 记录内容 | 频率 |
|------|---------|------|
| 训练开始 | 超参数 (model, loss, lr, batch_size, num_gpu, precision, dataset, peft, num_epoch) | 1次 |
| 训练循环 | loss, lr, grad_norm_backbone, grad_norm_classifier, update_ratio, adaface统计 | 每200步 |
| 每epoch评估 | 各benchmark summary分数 (LFW, CFP-FP, AgeDB, IJB-B/C, TinyFace等) | 每epoch |
| 最终评估 | final/ 前缀的最佳模型评估分数 | 训练结束 |

## 数据存储

- 格式：SQLite 数据库文件
- 位置：`{output_dir}/mlflow.db`
- 单文件，可直接复制/备份/分享

## 查看结果

### 启动 UI

```bash
# 查看某次训练的结果
mlflow ui \
    --backend-store-uri sqlite:///path/to/output_dir/mlflow.db \
    --host 0.0.0.0 --port 5000

# 示例
mlflow ui \
    --backend-store-uri sqlite:////data2/dataset_0213_rec/train_output/my_run_Jun-08_001/mlflow.db \
    --host 0.0.0.0 --port 5000
```

### 浏览器访问

```
http://<服务器IP>:5000
```

### UI 操作

1. 左侧 Experiments 栏选择实验名称
2. 点击某个 Run 查看：
   - Parameters：所有超参数
   - Metrics：各指标随 step 变化的曲线图
   - Artifacts：保存的文件
3. 勾选多个 Run → Compare：多实验对比图表

### 关闭 UI

```bash
# Ctrl+C 或
kill $(lsof -t -i:5000)
```

## 分享数据

### 方式1：发送 db 文件（最简单）

```bash
# 把 mlflow.db 发给对方，对方执行：
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

### 方式2：导出为 CSV

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///path/to/mlflow.db")
runs = mlflow.search_runs(experiment_ids=["1"])
runs.to_csv("experiment_results.csv", index=False)
```

### 方式3：内网共享服务（多人协作）

```bash
# 服务端 (一台机器常驻)
mlflow server \
    --backend-store-uri sqlite:///shared_mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 --port 5000

# 客户端代码改一行
mlflow.set_tracking_uri("http://服务器IP:5000")
```

## Demo 脚本

`mlflow_demo.py` 可独立运行，模拟两组训练实验写入数据，用于验证 MLflow 功能：

```bash
conda activate cvlface
python mlflow_demo.py
# 然后启动 UI 查看
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

## DDP 多卡安全

所有 MLflow 写操作仅在 rank 0 执行，不会有 SQLite 并发冲突。其他 rank 不调用任何 MLflow API。

## metric 命名注意

MLflow metric 名称不支持 `@` 字符，代码中已自动将 `@` 替换为 `_at_`（如 `tar@far1e-4` → `tar_at_far1e-4`）。

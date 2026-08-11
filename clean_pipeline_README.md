# RecordIO 全量特征提取流水线

该目录用于从阶段 5 backbone checkpoint 对训练集 RecordIO 做一次全量 TensorRT FP16
推理。输出保留稳定的样本索引和完整特征，供后续 SubCenter 分类器打分与数据清洗使用。

## 数据契约

输入：

- `train.rec` / `train.idx`：图片 RecordIO。
- `train.tsv`：三列无表头数据，依次为 `record_idx`、相对图片路径和 `label`。
- 阶段 5 checkpoint 中的 `model.pt` / `model.yaml`：只使用 backbone 提取特征。

`train.tsv` 只作为输入，不复制到输出目录。输出布局如下：

```text
/data1/zkj_work/subcenter/feats/
├── index/
│   └── part-000000.parquet
├── chunks/
│   └── features-000000.npy
├── failures/
│   └── part-000000.parquet
├── cache/
├── features.zarr/
├── index_manifest.json
├── manifest.json
├── _INDEX_SUCCESS.json
└── _SUCCESS.json
```

每个索引 Parquet 包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `row_idx` | UInt64 | 最终 `features.zarr/features` 的 0 基行号 |
| `record_idx` | UInt32 | `train.idx` / `train.rec` 的记录编号 |
| `path` | String | `train.tsv` 中的相对图片路径 |
| `label` | UInt32 | identity 分组 |
| `chunk_id` | UInt32 | 对应 NPY 分块编号 |
| `chunk_offset` | UInt32 | 样本在 NPY 分块内的行号 |

严格对齐关系：

```text
features.zarr/features[row_idx]
  == chunks/features-{chunk_id:06d}.npy[chunk_offset]
```

特征保存为模型原始 FP16 输出，维度为 512。不做水平翻转，不做 L2 归一化；后续与
SubCenter 分类器中心计算余弦相似度时再对特征和中心统一归一化。

图片解码失败、尺寸异常或 RecordIO label 与 `train.tsv` label 不一致时，该行特征写为
NaN，并在同编号的失败 Parquet 中记录错误。行不会被删除或前移，因此全局索引保持稳定。

## 执行

在 `try_subcenter` 目录使用 `cvlface` 环境：

```bash
conda run -n cvlface python clean_pipeline/extract_recordio_features_trt.py
```

默认参数为 8 张 GPU、每卡 batch 512、每卡 4 个解码 worker、每个 NPY 100,000 行，
输出到 `/data1/zkj_work/subcenter/feats`。全量 FP16 特征约占 35.4 GiB，索引 Parquet、
NPY 分块与最终 Zarr 会同时保留。

流水线可分阶段运行：

```bash
# 只将 train.tsv 转换为分块 Parquet
conda run -n cvlface python clean_pipeline/extract_recordio_features_trt.py --stage index

# 提取 NPY 分块
conda run -n cvlface python clean_pipeline/extract_recordio_features_trt.py --stage extract

# 校验全部分块并汇总 Zarr
conda run -n cvlface python clean_pipeline/extract_recordio_features_trt.py --stage bundle
```

小样本 smoke 测试必须使用独立输出目录，避免 manifest 与全量任务冲突：

```bash
conda run -n cvlface python clean_pipeline/extract_recordio_features_trt.py \
  --output-dir /tmp/subcenter_feature_smoke \
  --num-gpus 1 \
  --num-workers 0 \
  --batch-size 16 \
  --rows-per-chunk 8 \
  --max-samples 16
```

## 续跑与完整性

- manifest 绑定输入文件、模型、TensorRT engine、batch size 和分块参数；不一致时拒绝复用。
- 已完成且 shape/dtype 正确的 NPY 分块会被跳过。
- NPY 和 Parquet 先写临时文件，再原子重命名。
- Zarr 按 NPY 顺序流式写入，不会把全部特征加载到内存。
- 只有全部 NPY 存在且通过结构校验后，才写 `_SUCCESS.json`。
- 发现孤立临时文件或不完整失败记录时会停止，避免静默覆盖现场。

## 尾部两条修复

当前 `train.idx` 的图片范围是 `1..37084481`，但 `train.tsv` 只覆盖到
`37084479`。先生成两条修复材料，不修改现有特征：

```bash
conda run -n cvlface python clean_pipeline/repair_missing_recordio_tail.py
```

检查 `tail_repair/manifest.json`、`tail_features.npy` 和 `tail_index.parquet` 后，显式应用：

```bash
conda run -n cvlface python clean_pipeline/repair_missing_recordio_tail.py --apply
```

应用时会备份最后一个 NPY 和索引 Parquet，随后补齐 Zarr 并更新完成标记。该步骤支持从
“NPY/Parquet 已合并但 Zarr 尚未更新”的中断状态继续。

## 三子中心打分

特征补齐后，直接读取 Zarr，不再重复解码 RecordIO 或运行 backbone：

```bash
conda run -n cvlface python clean_pipeline/score_subcenters_from_zarr.py
```

默认使用 8 张 GPU，输出到 `/data1/zkj_work/subcenter/scores/scores/part-*.parquet`。
每行包含三个类内余弦分数、分配子中心、第一/第二相似度、margin 和特征范数。中心按
label 从阶段 5 的八个 `classifier_rank*.pt` 中定位，不计算全部类别 logits。

## Drop75 审计

打分完成后按 identity 统计主导子中心并生成候选清洗决策：

```bash
conda run -n cvlface python clean_pipeline/build_drop75_decisions.py
```

默认阈值为 `cos(75 degrees) = 0.2588190451`，输出：

- `identity_summary.parquet`：每类三个子中心的数量、主导中心和主导比例。
- `decisions.parquet`：逐样本主导中心相似度与 `keep_drop75`。
- `audit.json`：保留/删除数量、比例和相似度分位数。

该阶段只生成审计结果，不删除样本，也不修改原始 RecordIO。

# work_0605 训练流程说明

## 1. 数据打包 (图片 → RecordIO)

将散落在子目录中的图片打包为 RecordIO 格式，供训练使用。

### 数据目录结构要求

```
source_dir/
├── label_folder_1/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── label_folder_2/
│   └── ...
└── ...
```

每个子文件夹代表一个类别（label），文件夹名不限格式，会按自然排序映射为 0, 1, 2, ... 的数字标签。

### 打包脚本

**位置:** `cvlface/data_utils/recognition/training_data/bundle_images_into_rec_v2.py`

**特点:**
- 无 mxnet 依赖，纯 Python 实现
- 8 读线程并行预读 + 单线程顺序写入，NVMe 上约 13K-15K img/s
- 非 JPEG 图片自动转换为 JPEG (quality=100)
- 同时生成 train.tsv（训练时直接加载，跳过遍历 rec 文件）
- 与 mxnet RecordIO 格式完全兼容：无 header 记录，索引从 0 开始

### 调用方式

```bash
# 基本用法：打包到 source_dir 下（rec 文件和源图片同目录）
python bundle_images_into_rec_v2.py --source_dir /data1/dataset_0605/train

# 指定输出目录
python bundle_images_into_rec_v2.py --source_dir /data1/dataset_0605/train --save_dir /data1/dataset_0605/train_rec

# 打包后删除源图片
python bundle_images_into_rec_v2.py --source_dir /data1/dataset_0605/train --save_dir /data1/dataset_0605/train_rec --remove_images
```

### 输出文件

```
save_dir/
├── train.rec       # RecordIO 数据文件
├── train.idx       # 索引文件（idx → 字节偏移）
├── train.tsv       # 元数据（image_index \t label/filename \t label）
└── meta.json       # {"num_classes": N, "num_samples": M}
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--source_dir` | 是 | 包含标签子文件夹的图片根目录 |
| `--save_dir` | 否 | 输出目录，默认与 source_dir 相同 |
| `--remove_images` | 否 | 打包完成后删除源图片目录 |

---

## 2. 训练

### 前提

确保打包后的目录中包含 `train.rec`、`train.idx`。`train.tsv` 可选（没有时训练启动阶段会自动从 rec 生成，但多卡时会等待）。

### 启动命令示例

```bash
fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=s1_warmup_0605_v2 \
    $COMMON \
    models.freeze=True \
    pefts=configs/freeze.yaml trainers.batch_size=512 \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=5 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0 \
    trainers.skip_final_eval=True
```

### 多卡注意事项

- `train.tsv` 不存在时，只有 rank 0 会遍历 rec 文件生成，其余进程等待 barrier
- 建议提前用打包脚本生成 train.tsv，避免训练启动时 8 个进程重复遍历 3500 万条记录
- 如果之前多卡训练时产生了不完整的 train.tsv（多进程写冲突），需要先删除再重新生成

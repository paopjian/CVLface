# 数据格式对比训练方案

## 1. 打包

### RecordIO (推荐, 7-8 分钟)

```bash
conda run --no-capture-output -n cvlface python scripts/benchmark/bundle_rec_large.py
```

- 输入: `/data1/dataset_0605/train` (791K 类, 37M 图片)
- 输出: `/data1/dataset_0605/train_rec/train.rec` + `train.idx`
- 预计耗时: 7-8 分钟 (纯顺序写, 82K it/s)
- 预计磁盘: ~150GB

### LMDB (可选, 较慢)

```bash
conda run --no-capture-output -n cvlface python scripts/benchmark/bundle_lmdb_large.py
```

- 输出: `/data1/dataset_0605/train_lmdb/train.lmdb/`
- 预计耗时: 1-2 小时 (B+tree 写入开销)
- 预计磁盘: ~160GB
(cvlface) root@star:~/zhaokj/CVLface_folder/cvlface/research/recognition/code/work_0605# python scripts/benchmark/bundle_lmdb_large.py
Scanning /data1/dataset_0605/train ...
Using 64 workers to scan 791509 directories...
Scanning dirs: 100%|██████████████████████████████████████████████████████████████████████████████████████| 791509/791509 [00:43<00:00, 18400.86dir/s]
Found 37,084,481 images, 791,509 classes in 59.8s

[LMDB] Writing to /data1/dataset_0605/train_lmdb/train.lmdb
  map_size: 322 GB
  mode: writemap + 只最后 commit 一次
LMDB: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 37084481/37084481 [51:00<00:00, 12115.32it/s]
Committing (final sync)...
[LMDB] Done: 3063.4s (51.1 min), size: 322.1 GB
  速度: 12106 images/s

完成! LMDB 路径: /data1/dataset_0605/train_lmdb/train.lmdb

## 2. 对比训练 (阶段一, 跑 1 epoch 对比速度)

### 公共环境

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
conda activate cvlface

COMMON="trainers.num_gpu=7 trainers.num_workers=8 \
trainers.precision=bf16-mixed \
models=iresnet/configs/v1_ir101.yaml \
models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
data_augs=configs/basic_v1.yaml \
classifiers=configs/partial_fc_sample10.yaml classifiers.sample_rate=0.40 \
losses=configs/adaface.yaml \
evaluations=configs/val_20260605.yaml \
dataset.model_save_dir=/data1/dataset_0605/train_output"
```

### A) ImageFolder 原始格式 (baseline)

```bash
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=bench_s1_imgfolder \
    $COMMON \
    dataset=configs/dataset_0605_train.yaml \
    models.freeze=True \
    pefts=configs/freeze.yaml trainers.batch_size=512 \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=1 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0 \
    trainers.skip_final_eval=True
```
Epoch 0 | Step 10347 | Batch 10346 | Speed 26674 | LR 0.00800 | Loss 16.3750: 100%|█████████████████████████████| 10347/10347 [40:09<00:00,  7.37it/s]Epoch Time: 40.23 minsEpoch Time: 40.23 minsEpoch Time: 40.21 mins

### B) RecordIO 格式

```bash
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=bench_s1_rec \
    $COMMON \
    dataset=configs/dataset_0605_train_rec.yaml \
    models.freeze=True \
    pefts=configs/freeze.yaml trainers.batch_size=512 \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=1 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0 \
    trainers.skip_final_eval=True
```

### C) LMDB 格式 (打包完成后)

```bash
fabric run --devices=7 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=bench_s1_lmdb \
    $COMMON \
    dataset=configs/dataset_0605_train_lmdb.yaml \
    models.freeze=True \
    pefts=configs/freeze.yaml trainers.batch_size=512 \
    optims=configs/step_sgd.yaml \
    optims.lr=0.008 optims.num_epoch=1 "optims.lr_milestones=[2,4]" \
    optims.momentum=0.9 optims.weight_decay=0.0001 \
    optims.lr_lambda=0.3 optims.max_grad_norm=5.0 \
    trainers.skip_final_eval=True
```
poch 0 | Step 10347 | Batch 10346 | Speed 24589 | LR 0.00800 | Loss 16.3750: 100%|█████████████████████████████| 10347/10347 [29:36<00:00,  6.87it/s]Epoch Time: 29.86 minsEpoch Time: 29.86 mins
## 3. 对比观察点

- tqdm 里的 Speed (img/s)
- Epoch Time
- `nvidia-smi` 观察 GPU 利用率是否有卡掉到 70%
- wandb 里 train/loss 曲线是否一致 (验证训练正确性)

## 4. 代码改动说明

| 文件 | 改动 |
|------|------|
| `fabric/fabric.py` | DataLoader 加 `prefetch_factor=3` |
| `dataset/__init__.py` | 加 LMDB 检测: 如果 `train.lmdb/` 目录存在, 走 LMDBFaceDataset |
| `dataset/configs/dataset_0605_train_lmdb.yaml` | LMDB 版 dataset config |
| `dataset/configs/dataset_0605_train_rec.yaml` | RecordIO 版 dataset config |
| `scripts/benchmark/bundle_lmdb_large.py` | 大数据集 LMDB 打包脚本 |
| `scripts/benchmark/bundle_rec_large.py` | 大数据集 RecordIO 打包脚本 (推荐) |

## 5. 格式对比

| 指标 | ImageFolder | RecordIO | LMDB |
|------|-------------|----------|------|
| 打包时间 | 0 | **7-8 分钟** | 1-2 小时 |
| 磁盘占用 | ~150GB (37M 文件) | ~150GB (2 文件) | ~160GB (1 目录) |
| 读取方式 | 随机文件 open/read | seek + read 单文件 | mmap B+tree |
| 多进程竞争 | inode/dentry 压力大 | 单文件句柄竞争 | 无锁并发读 |
| 小数据集 (< RAM) | 最快 | 持平 | 持平 |
| 大数据集 (> RAM) | IOPS 瓶颈 | 较好 | 最好 |

## 6. 为什么 LMDB 打包慢

- LMDB 是事务型 B+tree 数据库, 每次 commit 需要 fsync
- 数据库越大, B+tree 深度增加, 脏页越多, fsync 越慢
- RecordIO 是纯顺序 append write, 没有索引维护开销

打包是一次性成本; 训练时读取性能才是核心。

# AGENTS.md — work_260922 工作区指南

人脸识别 (CVLFace) 训练/评估**性能优化**实验工作区。模型 iResNet-101 (AdaFace, WebFace12M, 512d), 硬件 7-8× RTX 4090。
本区由 `work_0605` 同步而来 (2026-09-22), 是当前活跃工作区; 文档全部为中文。

## 环境 (必须先设)

无系统 python, 一切用 conda env `cvlface`。启动前:

```bash
export CUDA_HOME=/root/anaconda3/envs/cvlface
export PATH=/root/anaconda3/envs/cvlface/bin:$PATH
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH  # 解决 scipy libstdc++ CXXABI 冲突
```

nvjpeg 管线另需 pip 包 `nvidia-nvjpeg-cu12` 与 `ninja`; CUDA 融合核需 nvcc (已装 cuda-nvcc 12.9)。
环境自检: `python portable_eval/check_env.py --compile-test`。

## 目录速览

| 路径 | 用途 |
|---|---|
| `train_opt.py` | 优化后的训练入口 (compile+channels_last+cudnn, +31%); `train.py`/`train5.py`/`try_train.py` 为旧入口 |
| `eval_all_trt_single.py` / `_launcher.py` | TRT fp16 多卡评估 (无 fabric/NCCL, 主进程建 engine + 多进程提取) — 生产评估主力 |
| `eval_all_torch_single.py` / `_launcher.py` | torch 多卡评估 (fabric DDP, 支持 `--compile --compile_mode max-autotune`) |
| `external_torch_eval.py` | 训练 epoch 末外挂评估 (可切 TRT 后端, 子进程拉起 + JSON 回读) |
| `evaluations/` | 评估器 + `cluster_utils.py` (**v7 CUDA 融合核**, 唯一相似度引擎, v6 已删) + `nvjpeg_pipeline.py` + 3 个 `.cu`; configs 里 `val_20260922`/`test_20260922` 是现行评估配置 |
| `portable_eval/` | 零仓库依赖的可迁移评测包 (nvjpeg 双流 + TRT + 移植版融合核), 见其 README |
| `bench_260922/` | 本轮基准脚本/结果/engine (`engines/adaface_ir101_bs512_fp16.engine`), 总结在 `REPORT_260922.md` |
| `opt_eval/` | 历史优化实验 (tensorrt / sim_matrix / nccl 等) |
| `docs/` | 专题文档 (TRT 优化、tinyface 死锁修复等) |
| 计划*.md, 流程.txt, eval_acceleration.md | 根目录中文计划/流程/结论文档, 改动前先读对应主题 |

配置系统: hydra + omegaconf, `base.yaml` 定 defaults, 各模块 `configs/` 子目录; 命令行覆盖如 `models=iresnet/configs/v1_ir101.yaml`。实验输出在仓库级 `research/recognition/experiments/<task>/` (本区之外)。

## 常用命令

```bash
# 训练 (8卡 DDP)
fabric run --strategy=ddp --devices=8 --precision="bf16-mixed" train_opt.py \
    trainers.prefix=<exp> trainers.num_gpu=8 trainers.batch_size=512 \
    models=iresnet/configs/v1_ir101.yaml dataset=configs/<data>.yaml \
    data_augs=configs/gridsample_v2_numpy.yaml classifiers=configs/partial_fc_sample10.yaml \
    losses=configs/adaface.yaml evaluations=configs/val_20260922.yaml

# TRT 评估 (批量 checkpoint)
python eval_all_trt_launcher.py --num_gpu 8 --eval_config_name test_20260922 \
    --ckpt_dir <ckpt目录> --project_name <proj> --name <run> --timeout_minutes 90

# portable_eval 全链路
python portable_eval/run_eval.py --engine bench_260922/engines/adaface_ir101_bs512_fp16.engine \
    --data <rec或folder数据> --source rec --pipeline nvjpeg --gpus 8 --bs 512 --tag <tag>

# 单测 (无正式测试框架, 均为可独立运行脚本)
python bench_260922/test_nvjpeg_pipeline.py      # nvjpeg 管线单测
python bench_260922/test_external_trt_eval.py    # 外挂 TRT 评估链路
```

## 关键约束 (load-bearing, 勿破坏)

1. **nvjpeg 永不做部分批调用**: `nvjpegDecodeBatched` 收到少于 init batch 的批会段错误/句柄损坏 (与图片内容无关)。尾批必须补满 (`n_valid` 截取) 或跳过。`evaluations/nvjpeg_pipeline.py` 与 `portable_eval` 均内建此约束; 自定义管线必须遵守。
2. **TRT IO binding 按 mode 识别**: 用 `get_tensor_mode()==TensorIOMode.INPUT` 找输入, 不能假设 index 0/1 顺序 (TRT 10/11), 否则特征全错 (cos 可低至 0.7)。
3. **TRT 11 精度**: 无 `BuilderFlag.FP16`, 导出 fp16 ONNX + STRONGLY_TYPED 网络即得 fp16 engine。
4. **相似度矩阵只有 v7**: `evaluations/cluster_utils.py` 的 `get_sim_matrix_large_scale_v7` (fp16 GEMM + CUDA 双桶直方图, hist@2000 bins)。v6 入口已删除, 不要再引入。
5. **nvjpeg 仅支持 JPEG**: PNG 数据集 (cplfw/calfw/agedb 等 HF 开源集) 自动降级 cv2; 降级必须**全体 rank 集体表决** (all_reduce), 单 rank 独行会导致 barrier 错位死锁。
6. **torch.compile 必须在 `fabric.setup(model)` 之前调用**, 否则 DDP wrapper 导致 graph break。
7. **DataLoader worker 用 fork**: rec 分支曾因全局 spawn 上下文 pickle `BufferedReader` 失败 (8 worker 全挂且静默), 已改 `multiprocessing_context='fork'`。
8. **TRT engine 缓存默认关闭** (每 epoch 新权重, 缓存命中趋零还占盘); 仅显式传 `--engine_cache_dir` 才落盘。冷构建需 GPU0 约 5GB 空闲显存。
9. **RecSource npy offsets 缓存** (`train.offsets.npy`) 的 tmp 文件名必须 pid 唯一, 否则并发首开竞态互相 rename。
10. **口径差异**: 生产 evaluator 是 normal+flip 两遍取平均; portable_eval 单遍无 flip — TPIR 有可预期小差, 交叉验证时以此解释, 不是 bug。
11. mxnet 已全部移除: RecordIO 读取走纯 Python `dataset/recordio_reader.py` (fork-safe, PID 变化自动重开), 不要重新引入 mxnet。
12. **THP 同步规整死窗口**: 进程内评估的 GB 级 CPU 大数组操作 (sklearn normalize 等) 在 THP `defrag=madvise` + 内存碎片化时触发内核同步 direct compaction, 吞吐跌 1~2 个量级, 表现为评估期间全 GPU 空转且耗时无规律波动 (7 卡机 2026-09-23 A/B 验证: defrag=never 后 compact_stall 冻结、单集 656s→164s)。机器级修复 `echo never > /sys/kernel/mm/transparent_hugepage/defrag` (已持久化于 /etc/tmpfiles.d/thp-defrag.conf); `train_opt.py` 启动时另有进程级 prctl(PR_SET_THP_DISABLE) 保险 (`CVLFACE_DISABLE_THP=0` 关闭)。换新机器先查 `grep compact_stall /proc/vmstat` 是否持续增长。
13. **断点续训 wandb 重连**: `train_opt.py` 的 `resolve_wandb_run` 按 CLI > checkpoint 存档 > 旧 run 目录解析 > 新生成的优先级定 run id (WandbLogger 内部 resume="allow"); 重启训练**必须**带 `trainers.resume` 才能重连同一 wandb run, 否则新开 run。

## Git

仓库根在 **4 级之上** (`/root/zhaokj/CVLface`)。本工作区已入库跟踪 (main 分支, 2026-09-22 起; bench 产物/权重/运行日志经 .gitignore 排除)。
兄弟目录 `work_0605` 亦有跟踪。远端 origin (github paopjian/CVLface) 在本机无 HTTPS 凭据, push 需在有凭据的终端执行。

## 文档优先级

改动评估链路前先读: `README.md` (全流程优化总结) → `eval_acceleration.md` (TRT/refit/精度) → `bench_260922/REPORT_260922.md` (最新基线与剩余优化点, 附4 有逐项审计) → `portable_eval/README.md` (nvjpeg 坑的完整溯源)。数据集口径与打包见 `流程.txt`、`计划*.md`。

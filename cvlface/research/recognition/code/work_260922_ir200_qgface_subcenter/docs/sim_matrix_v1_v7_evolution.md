# get_sim_matrix_large_scale v1–v7 迭代总结

> 分析日期: 2026-09-23
> 分析对象: `evaluations/cluster_utils.py` (静态阅读 + git 历史, 未运行代码)
> 关联文档: `docs/sim_matrix_optimization.md` (v4→v5 时期的瓶颈分析心得), `bench_260922/REPORT_260922.md`

## 问题背景

给定 N 个 L2 归一化特征与身份标签, 统计全相似度矩阵 (约 N²/2 对) 中**正样本对 (同 ID) / 负样本对 (跨 ID) 的相似度直方图**, 配合 `compute_tpir_from_hist` 计算 TPIR@FAR 类指标, 并可选择性收集越过阈值的样本对 (neg 高分对 / pos 低分对)。

核心矛盾始终是: **N² 规模计算量 × 有限的 GPU 显存 / CPU 内存 / PCIe 带宽**。

版本说明: 文件中不存在名为 `v1`/`v2` 的函数, 仓库最早入库 (706f397, work_0605 时期) 时就已带 v3/v4。所谓 v1/v2 是 v3 之前的两代前身函数, 至今仍在文件里。

## 谱系总览

| 版本 | 入口/位置 | 一句话定位 |
|---|---|---|
| v1 (前身) | `get_similarity_matrix` | 单卡, 物化全部 O(N²) 分数到 CPU |
| v2 (前身) | `get_sim_matrix_batch_balanced(_silent)` | 多卡静态贪心调度 + topk/threshold 模式, 仍物化分数 |
| v3 | `get_sim_matrix_large_scale_v3` | 直方图化: 内存 O(N²)→O(bins), PairCollector 收集阈值对 |
| v4 | `get_sim_matrix_large_scale_v4` | 动态任务池, 运行时负载均衡消长尾 |
| v5 | `get_sim_matrix_large_scale_v5` | masked_fill_(NaN)+histc 单趟, 消 bool indexing; fp16 GEMM |
| v6 | `get_sim_matrix_large_scale_v6` | torch 原语集大成: shard 分片/行缓存/零拷贝锁页/跳正检查/守恒补偿 |
| v7 | `get_sim_matrix_large_scale_v7` | 自定义 CUDA 融合核, fp16 直读单遍完成直方图+样本对提取 (**当前唯一生产引擎**) |

## 各版本详解

### v1 (前身) — `get_similarity_matrix` (cluster_utils.py:441)

- 单卡 `cuda:0`, 按 10240 (2048×5) 大小的块双层循环走上三角。
- 每块 matmul 后用 boolean indexing (`sim_block[mask]`) 展平, 把**全部分数 `.half().cpu()` 物化**成 pos/neg 两个大数组。
- 找高相似负对时逐个 `.item()` 循环 (极慢)。
- 瓶颈: O(N²) 分数搬回 CPU 内存 + 单卡。

### v2 (前身) — `get_sim_matrix_batch_balanced(_silent)` (cluster_utils.py:798)

- `TriangularBlockScheduler` (cluster_utils.py:507): 上三角块按实际面积算工作量 (对角块也按完整矩形), 贪心大块优先**静态分配**给各 GPU。
- `ThreadPoolExecutor` 每 GPU 一线程; 特征预先 `pin_memory` 加速 H2D。
- core 函数 (cluster_utils.py:624) 多输出模式:
  - `topk`: GPU 上增量维护 top-k 负样本, 免全量排序回传;
  - `threshold`: GPU 批量索引映射取对 (替代逐 `.item()`);
  - `return_stats_only` / `return_pairs_only`: 只要计数/只要样本对。
- silent 变体 (cluster_utils.py:1000): 关打印 + 用后 `empty_cache` 清显存。
- 仍物化全部分数。

### v3 — `get_sim_matrix_large_scale_v3` (cluster_utils.py:1376)

**决定性转折: 直方图化。**

- 每块在 GPU 上直接 `torch.histc` (20M bins), 各卡只回传 `(pos_hist, neg_hist)`, 主线程逐 bin 相加——内存从 O(N²) 降到 O(bins)。
- `PairCollector` (cluster_utils.py:1193): 线程安全、带上限的阈值样本对收集; 支持单模式 (一种类型) / 双模式 (pos/neg 各自阈值与方向)。
- `memory_mode`: `high_performance` 特征常驻显存 (OOM 自动降级), `low_memory` pinned CPU 按块 H2D。
- 调度仍用静态 `TriangularBlockScheduler`。

### v4 — `get_sim_matrix_large_scale_v4` (cluster_utils.py:1681)

- `DynamicBlockPool` (cluster_utils.py:1502) **动态任务池**: 块不预绑定 GPU, 各卡运行时取 (首取 1%, 后续每次 0.5%), 块按工作量降序大块优先, 快卡自动多干——消除卡速不均时的静态分配长尾。
- 显式**不做 `gc.collect()`** (大规模计算后遍历对象图可达数百秒, 踩过的坑), 改为逐卡 `empty_cache`。

### v5 — `get_sim_matrix_large_scale_v5` (cluster_utils.py:2005)

计算原语优化 (详见 `docs/sim_matrix_optimization.md`):

- 对角块下三角与"非同 ID"位置改 `masked_fill_(NaN)` + `histc` (NaN 自动丢弃): 一块矩阵只做**两次直方图、neg = full − pos**, 消除 boolean indexing 的动态 shape 分配与隐式同步 (一个 10240² 块原来要触发 4 次动态分配)。
- `precision='fp16'`: 半精度 matmul 上 Tensor Core (4090 上 TF32 165 TFLOPS vs FP32 82 TFLOPS), 提示 hist_bins 降到 2000。
- 代价: collect 模式需 `sim.clone()` (histc 是 in-place)。
- 峰值显存更低 (不再有 flat_sim/flat_labels 中间张量)。

### v6 — `get_sim_matrix_large_scale_v6` (cluster_utils.py:2432)

torch 原语时代的集大成, 一次加了一整组手段:

- **调度**: 单 tile 粒度 `queue.Queue` + 裸线程; tile 顺序"行块序 + 行内宽优先" (行缓存友好 & 大块先跑); block_size 默认 16384 (更大 GEMM)。
- **传输**:
  - `pin_inplace`: `cudaHostRegister` 原地锁定 numpy 缓冲区, 零拷贝 (省 `pin_memory()` 整份副本);
  - `row_cache`: 同一行块扫多个列块只 H2D 一次;
  - `memory_mode='shard'`: 每卡常驻 1/G 的**列块**分片, tile 按列归属路由 (对角 tile 归行块卡), 把 low_memory 的 O(nb²) 次列块 PCIe 传输降到线性, 面向千万级 N。
- **计算**:
  - `skip_pos_check` (`_pos_tile_flags`, cluster_utils.py:2177): 静态预标哪些 tile 含同 ID 对 (对角块内同 ID ≥2; 非对角跨块标记区间, 保守不漏标), 无正样本 tile 整块跳过 `label_eq` 身份比对——评估数据里绝大多数 tile 无正样本;
  - `skip_clamp` 守恒补偿: 不做全量 clamp (省一趟读写), 先解析算出本 tile 应有对数 n_valid/n_same, histc 丢掉的越界值补回边界 bin (语义"越界算 ±1");
  - `precision` 三档 fp32/tf32/fp16 (tf32 默认, 实测快 ~12% 且 0 bin 差);
  - `hist_method` 备选 bincount (量化 + GPU bincount);
  - pair 收集直接用 in-place 修改前的 sim, 省掉 v5 的整份 clone。

### v7 — `get_sim_matrix_large_scale_v7` (cluster_utils.py:2685, 当前唯一生产引擎)

走出 torch 原语, **自定义 CUDA 融合核单遍完成**:

- fp16 GEMM 结果**不转 fp32 直接喂核**。两个核文件:
  - `cuda_histpn_f16.cu` (55 号): 双桶直方图核, fp16 直读 + 寄存器转 fp32 + smem 双桶私有化 + 步进增量坐标;
  - `cuda_histpn_he_fused.cu` (`fused_he_pn`): 55+54 融合, **单遍一次读完成双桶直方图 + 双阈值样本对提取** (neg: 跨 ID 且 ≥thr_neg; pos: 同 ID 且 ≤thr_pos)——collect 模式从"两遍"变"单遍"。
- 身份比较降为 **int32 身份码** (`np.unique` 逆映射) 等值比较, 省带宽; 对角 tile 严格上三角在核内 `is_diag` 处理。
- 数据布局继承 v6: 列分片常驻 (bj % G 路由) + 行块缓存 + `cudaHostRegister`。
- 每卡预分配 CAP=3200 万对的 `(i,j,score)` 缓冲 (int32/fp32), 核内 atomicAdd 写入, 超缓冲即 assert; 末尾**守恒校验** (直方图总数 vs 解析对数, 差 >1e-6 比例直接 assert)。
- 统一三模式入口: A hist-only / B 样本对 / C hist+pairs。
- 默认值随精度改: `hist_bins=2000` (fp16 分辨率极限 ~0.001), `block_size=16384`。
- 演进提交: `1393a02` 双桶核接入 cv4 TPIR → `bd6ac7b` 三模式统一入口 + 两处 bug 修复 → `6fd9415` collect 单遍化融合核。

## 演进主线

| 维度 | 路径 |
|---|---|
| 输出表示 | 物化 O(N²) 分数 → GPU 直方图 O(bins) (v3) |
| 调度 | 静态贪心 (v2) → 动态池 (v4) → tile 队列 + 列分片路由 (v6/v7) |
| 单块计算 | bool indexing (v1–v3) → NaN+histc 单趟 (v5) → 融合 CUDA 核单遍 (v7) |
| 精度 | fp32 → fp16/tf32 GEMM (v5 起) → fp16 直读核 (v7) |
| 传输 | pin_memory → cudaHostRegister 零拷贝 + 行缓存 (v6) → 列分片常驻 (v6 shard, v7 内建) |

## 当前引用状态

- 生产评估器已**全部统一 v7**: `custom_verification_evaluator.py:1021`, `custom_ijbbc_evaluator.py:220`。
- **合并评估已切 v7** (2026-09-23): `evaluations/__init__.py` 的 `run_combined_evaluations` (多源 embedding 拼接重算 TPIR) 原先硬编码调 v4 (`block_size=2048*2`, 20M bins), 已改为与 type4 评估器同口径的 v7 (`block_size=16384, hist_bins=2000, hist_range=(-1,1)`, `EVAL_NUM_GPUS` 覆盖 GPU 数默认 8), 且 `compute_tpir_from_hist` 显式传 `hist_bins/hist_range` (其默认 20M bins, 不传会阈值映射错位)。口径变化: fp32→fp16 GEMM, TPIR 可能有 fp16 分辨率 (~0.001) 内的微小差异。
- v6 函数体保留: 生产链路无调用方 (`9e0ae49` 删除 `--v6_matrix` 调试入口), 但 `opt_eval/tensorrt/` 下 5 个 v6-vs-v7 A/B 基准脚本 (`cv4_v7_poc.py`, `cv4_v7_collect_poc.py`, `ijbc_v6hist_poc.py`, `ijbc_bins2000_poc.py`, `ijbc_twopass_poc.py`) 仍 import 它作对比基线, 删除会破坏这些历史实验, 故保留。

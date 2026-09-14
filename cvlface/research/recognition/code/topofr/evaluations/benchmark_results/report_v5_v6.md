# v5 vs v6 相似度矩阵速度对比评估报告

- 日期：2026-09-12
- 模型：s4_0618（ir101 / 512 维，`/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/s4_0618`）
- 硬件：7 × RTX 4090（24GB），driver 575.57.08，triton 3.7.0，torch 2.12.0+cu126
- 环境：cvlface（Python 3.12.13），需 `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`
- 配置：`test_20260605.yaml` / `val_20260605.yaml`（仅 `custom_verification4` 类型走 v5/v6 直方图路径；ijbbc/ijbc_custom 走 topk、tinyface/verification 不走大规模 sim matrix，不受 v6 影响）
- 明细数据：`bench_v5_v6.parquet`（本目录）

## 1. 端到端整轮耗时（single 实测）

| 链路 | test_20260605 | val_20260605 |
|---|---|---|
| TRT + v5 | 33.0 min | 8.9 min |
| TRT + v6 | 35.8 min | 8.9 min |
| Torch + v5 | 87.6 min | 9.0 min |
| Torch + v6 | 93.0 min | 10.2 min |

TRT engine 构建约 60s/轮；torch 链路为 DDP（bf16-mixed）+ evaluator 全流程。

## 2. sim matrix 耗时明细（v5 vs v6，各自链路现网参数）

**TRT 链路**（`block=32768, hist_bins=2000, fp16`，两版本同参数）：

| 数据集 | N | v5 (s) | v6 (s) | v6/v5 |
|---|---|---|---|---|
| test_3t | 4,284,857 | 65.3 | 119.2 | 1.83× 慢 |
| test_enhance | 203,276 | 1.3 | 0.9 | 1.44× 快 |
| test_glint | 6,820,523 | 139.9 | 293.8 | 2.10× 慢 |
| test_1201 | 2,665,669 | 28.5 | 46.4 | 1.63× 慢 |
| val_3t | 860,159 | 6.2 | 11.0 | 1.77× 慢 |
| val_enhance | 41,058 | 0.3 | 0.2 | 1.5× 快 |
| val_glint | 1,365,613 | 8.5 | 13.2 | 1.55× 慢 |

**Torch 链路**（`block=8192`；v5=fp32 + bins=20M（v5 默认），v6=tf32 + bins=200k + triton epilogue（v6 默认设计值））：

| 数据集 | N | v5 (s) | v6 (s) | v6/v5 |
|---|---|---|---|---|
| test_3t | 4,284,857 | 166.0 | 634.2 | 3.82× 慢 |
| test_enhance | 203,276 | 5.0 | 2.4 | 2.1× 快 |
| test_glint | 6,820,523 | 346.4 | 1609.5 | 4.65× 慢 |
| test_1201 | 2,665,669 | 88.0 | 249.5 | 2.83× 慢 |
| val_3t | 860,159 | 9.3 | 28.8 | 3.10× 慢 |
| val_enhance | 41,058 | 1.7 | 0.9 | 1.9× 快 |
| val_glint | 1,365,613 | 16.9 | 66.4 | 3.93× 慢 |

规律：**N ≤ 20 万时 v6 略快；N 越大 v6 越慢（2~4.6×）**。

## 3. 特征提取：Torch vs TRT（与 v5/v6 无关）

test yaml（约 1400 万张，4 个 custom_verification4 集）：
- TRT：约 1027s（17.1 min）
- Torch DDP bf16：约 2568s（42.8 min）→ **TRT 快约 2.5×**

## 4. v6 配置扫描（定位根因，N=2,665,669 真实特征，3.55e12 对）

| 配置 | 耗时 | 吞吐（对/s） |
|---|---|---|
| v5 fp16 bins=2k blk=32k（TRT 现网） | 26.6s | 1.34e11 |
| v6 fp16 bins=2k blk=32k **epi=triton** | 48.0s | 7.40e10 |
| **v6 fp16 bins=2k blk=32k epi=torch** | **17.1s** | **2.07e11（最快）** |
| v6 fp16 bins=200k blk=32k epi=triton | 249.1s | 1.43e10 |
| v6 tf32 bins=200k blk=16k epi=triton（生产配置） | 246.7s | 1.44e10 |
| v6 tf32 bins=200k blk=16k epi=torch | 24.8s | 1.44e11 |
| v5 fp32 bins=20M blk=8k（evaluator 现网） | 53.8s | 6.61e10 |
| v6 tf32 bins=20M blk=8k epi=triton | 33.8s* | 1.05e11 |
| v6 tf32 bins=20M blk=8k epi=torch | 34.6s | 1.03e11 |

\* 注：bins=20M 并非 v6 默认值（v6 默认 200k），此行为额外构造的补充配置；20M bins 下 triton kernel 编译失败（`tl.zeros([BINS])` 超限），7 卡自动回退 torch epilogue —— 实质为 torch 路径耗时，与下一行 epi=torch 基本一致可互证。

所有配置的 pos/neg 直方图总数完全一致（pos=218,489,870 / neg=3,552,675,786,076），**v5/v6 正确性对拍通过**。

## 5. 结论

1. **triton 融合 epilogue 在本机为负优化**（RTX 4090 + triton 3.7.0 + torch 2.12.0）：Torch 链路 v6 即为原设计配置（tf32 + 200k bins），其中 bins_pow2=262144 > 8192 走逐元素 atomic 路径，同参数扫描显示该路径比 torch epilogue 慢约 10×（246.7s vs 24.8s）—— 这正是 Torch 链路 v6 慢 3.4~4.1× 的主因；TRT 链路（2k bins）走 tl.histogram 共享内存路径，同样慢于 torch epilogue（48.0s vs 17.1s）。
2. **源码注释中 "epilogue 17.65→6.37ms（2.77×）" 不可复现，且生产从未用过 triton**：(a) 重跑 `工具/eval_triton_epilogue.py`，torch 侧 17.65ms/tile 逐位复现，triton 侧实测 22.51ms/tile（0.78×，负优化）；(b) 时间线证明生产 2h05m（日志 9/7 21:52）早于 triton 集成进 v6（源码 9/9 03:17）与微基准脚本创建（9/8 14:24），生产速率反推每 tile ~18ms 非 GEMM 时间，与 torch epilogue 17.65ms 吻合 —— 生产跑的就是 torch epilogue；(c) 生产 8.0 脚本传 `collect_pairs_config`，按当前 v6 逻辑也会强制回退 torch epilogue。
3. **v6 原设计配置（tf32 + 200k bins）不宜直接替换现网 v5**：Torch 链路（即该配置，仅 block 8192 vs 生产 16384）大数据集慢 2.8~4.6×，TRT 链路慢 1.6~2.1×；仅小数据集（≤20 万）v6 略快。
4. **v6 调优后可反超 v5**：`epilogue='torch' + precision='fp16' + block_size=32768 + hist_bins=2000` 达 2.07e11 对/s，比同参数 v5 快 1.55×，比 evaluator 现网 v5 参数（fp32/20M/8k）快 3.1×。注意 bins 减小会降低 TPIR 阈值分辨率（2k bins 粒度 1e-3）。
5. **特征提取选 TRT**：大规模特征提取 TRT fp16 比 Torch DDP bf16 快约 2.5×。
6. IJBC evaluator（topk 路径，与 v5/v6 无关）在 torch 链路两轮间波动较大（1534s vs 303s），与环境状态有关，不影响上述结论。

## 6. 建议

- 现网维持 v5；若切换 v6，必须 `epilogue='torch'` 并搭配 fp16 + 大 block（32768）+ 小 bins（≤2000），可获 ~1.5× sim 加速（相对 TRT 现网参数）。
- 若坚持 triton 路径，需先在目标机复测 `工具/eval_triton_epilogue.py` 微基准；本机该路径不稳定（编译失败/性能倒挂）。
- v6 的 `memory_mode='shard'`（生产 3342 万图验证的模式）未在本次评估范围（评估集最大 682 万），超大规模挖掘场景仍建议 shard。

## 附：代码变更

- 新增 `evaluations/cluster_utils_v6.py`（自 `cv_datapipe6/4.3.clean_dataset/cluster_utils_v6.py` 原样移植）
- `eval_all_trt_single.py`：新增 `--sim_version {v5,v6}`；`compute_metric_type4` 按版本分发并打印 sim 耗时
- `evaluations/custom_verification_evaluator.py`：type='4' 支持 `SIM_VERSION`/`EVAL_NUM_GPUS` 环境变量（默认 v5/8，原行为不变）
- 新增 `evaluations/configs/bench_smoke.yaml`（冒烟配置）、`evaluations/benchmark_results/`（结果数据）

## 7. 补充实验：hist-only 对照矩阵（9/13，20% 数据 666 万行，不收集样本对）

同一份排序数据（与三阶段 123.3s 那次完全同源），block=16384, bins=200k，7 卡：

| # | 引擎 | 输入 | 精度/epilogue | 耗时 | 吞吐(对/s) | pos 总数 vs 真值 |
|---|---|---|---|---|---|---|
| 1 | v5 整函数 | 未排序 | fp32 | 239.5s | 9.27e10 | -50,938（丢 sim>1） |
| 2 | v5 整函数 | **排序** | fp32 | 231.8s | 9.58e10 | -50,938 |
| 3 | **v6 整函数** | **排序** | tf32 + **torch** epi | **146.0s** | **1.52e11** | **精确** |
| 4 | v6 整函数 | 排序 | tf32 + triton epi | 1531.9s | 1.45e10 | 精确（但慢 10×） |
| 5 | v6 整函数 | 未排序 | tf32 + torch epi | 202.2s | 1.10e11 | 精确 |

结论：
1. **排序对 v5 无效**（239.5→231.8s，3%）：v5 每 tile 无条件身份比对 + 2 次 histc，算法结构无法利用排序信息。
2. **排序让 v6 整函数快 38%**（202.2→146.0s）：排序后 skip_pos_check 预标从"几乎全部 tile"降为 **806/83,028（0.97%）**，99% tile 跳过身份比对。
3. **triton epilogue 负优化再次确认**：同配置下 1531.9s vs 146.0s（10.5× 慢）。
4. **正确性**：v6 三个配置 pos 总数全部精确等于真值；v5 两种布局固定丢 50,938 个对（sim>1 被 histc range 丢弃，恰为同 ID 对集中的区间）。
5. "之前 v6 比 v5 慢"的完整归因：之前两条链路的 v6 都默认带 triton epilogue（负优化 10×）+ 输入未排序（skip 失效）；同为 torch epilogue 时 v6 整函数其实比 v5 快（配置 5 vs 1），排序后更快（配置 3），三阶段最快（123.3s 含收集）。

## 8. 补充实验：fp16 精度对照（9/13，同 20% 排序数据，hist-only）

| # | 配置 | 耗时 | 吞吐(对/s) | pos 总数 |
|---|---|---|---|---|
| a | v5 fp16 blk=16384 bins=200k（对齐矩阵） | 207.3s | 1.07e11 | 精确 |
| b | v5 fp16 **blk=32768 bins=2k（TRT 现网参数）** | 133.8s | 1.66e11 | 精确 |
| c | v6 fp16 blk=16384 bins=200k torch epi | 144.2s | 1.54e11 | 精确 |
| d | **v6 fp16 blk=32768 bins=2k torch epi** | **98.8s** | **2.25e11** | **精确** |

结论：
1. **v6 fp16 + torch epilogue + 现网参数（blk=32768, bins=2000）是最快形态**：98.8s / 2.25e11 对/s，比 v5 最快形态（b, 133.8s）快 35%，比 v5 fp32 矩阵形态（231.8s）快 2.35×，且直方图精确。
2. v5 的 fp16 提速（231.8→133.8s，42%）主要来自现网大 block + 小 bins 参数，fp16 本身在同参数下贡献约 12%（231.8→207.3s）。
3. fp16 对 v6 几乎无增益（144.2 vs 146.0s）：v6 的瓶颈在直方图 epilogue 而非 GEMM 精度。
4. 本次 fp16 下 v5 的 pos 总数精确（未复现 fp32 的 -50,938 丢失）；fp16 特征舍入可能使点积不再越过 1.0。丢失缺陷在 fp32 下稳定复现，使用 v5 fp32 时需注意。
5. 场景化推荐：只算直方图 → v6 fp16 torch epi + blk=32768 + bins=2000；需要挖掘正负样本对 → sorted 三阶段（收集与直方图 fused，123.3s@20% 含收集）。

## 9. triton epilogue 减速的机制实锤（寄存器溢出诊断，9/13）

编译产物诊断（triton 3.7.0，RTX 4090，num_warps=8）：

| kernel 变体 | n_regs | n_spills |
|---|---|---|
| 掩码 + POSHIST（主路径，bins=2k/200k） | **255（打满上限）** | **358（溢出）** |
| 无掩码部分变体 | 255 / 240 / 123 | 30 / 0 / 0 |
| 最轻分支（atomic 无掩码单直方图） | 123 | 0 |

机制解释：
1. kernel 内 hn/hp 双直方图累加器（bins_pow2 int32 向量）+ BR×BC sim 块缓冲 + 身份比较矩阵，寄存器需求超过每线程 255 上限 → **358 次溢出到 local memory（实为全局显存）**，每个热循环元素的访存延迟放大百倍以上。
2. bins>8192 时 USE_SHARED=0 退化为逐元素 global atomic_add（PyTorch histc 从不这么做——它总是共享内存聚合后归约）。
3. 旁证：sorted 版 neg 热循环（无掩码单直方图轻分支，零溢出）实测 733 it/s×7 卡 ≈ 9.5ms/tile 含 GEMM —— 轻分支本身是快的。

**最终结论的适用范围**：对本机环境（triton 3.7.0 + torch 2.12.0 + RTX 4090）+ 当前 kernel 代码，"triton epilogue 负优化"是最终结论（微基准两轮复测 + 端到端多配置 + 生产日志交叉验证 + 寄存器溢出实锤，四重证据）。但它不是"triton 不行"的普遍结论：kernel 存在明确修复路径（减小 BR×BC 块、POSHIST 拆分为独立 pass、直方图分桶降 BINS 向量宽度、调 num_warps/num_stages），修好后未必不能反超 torch.histc；在其他 triton 版本/其他机器上结果也可能不同。

## 10. 修复实验：v6 triton 路径的轻重分支区分（9/13）

确认：v6 整函数的 triton 调用对所有 tile 无条件传 ids（全部特化为溢出的重变体），而其 torch 路径反而有 `if has_pos` 区分 —— triton 路径漏掉了预标信息的利用。

修复（cluster_utils_v6.py，已备份 .bak_0913）：`use_ids = has_pos or is_diag`，无正样本且非对角的 tile（99.8%）以 `ids_row=None, do_poshist=False` 调用（轻变体，零溢出），pos 贡献恒 0。

实测（20% 排序数据，hist-only）：

| 配置 | 修复前 | 修复后 |
|---|---|---|
| v6 tf32 blk=16384 bins=200k triton | 1531.9s | **140.7s（10.9×）**，反超 torch epi（146.0s） |
| v6 fp16 blk=32768 bins=2k triton | — | 130.0s（torch epi 98.8s 仍更快） |

正确性：小规模与 torch epi 逐 bin 一致；20% 数据 pos/neg 总数精确等于真值。

最终格局（20% 排序数据，hist-only，快→慢）：
1. **v6 fp16 + torch epi + blk=32768 + bins=2k：98.8s（2.25e11 对/s）** ← 全局最优
2. v6 fp16 + triton(修复) + 现网参数：130.0s
3. v5 fp16 + 现网参数：133.8s
4. v6 tf32 + triton(修复)：140.7s（生产默认 bins 下 triton 修复后反超 torch）
5. v6 tf32 + torch epi：146.0s
6. v5 fp32：231.8~239.5s
7. v6 tf32 + triton（修复前）：1531.9s

## 11. 最终评估：TRT 链路全量 test/val 的 v6 配置矩阵（9/13）

条件：TRT fp16 提特征（每轮一次，6 个 v6 配置循环复用），bins=2000，block=32768，
v5 对照为同链路同参数的历史数据（fp16）。正确性：**全部 24 组（4 evaluator × 6 配置）pos 直方图总数与真值差 = 0**。

### test yaml（4 个 custom_verification4，共 3.60e13 对）sim 合计

| 配置 | 合计耗时 | 吞吐 | vs v5 |
|---|---|---|---|
| **v6 fp16 + torch + skip_clamp** | **116.7s** | **3.09e11** | **快 2.01×** |
| v6 fp16 + shard + torch | 153.8s | 2.34e11 | 快 1.53× |
| v6 tf32 + torch | 158.7s | 2.27e11 | 快 1.48× |
| v6 fp16 + torch（默认 clamp） | 162.7s | 2.21e11 | 快 1.44× |
| v6 tf32 + triton(修复) | 208.5s | 1.73e11 | 快 1.13× |
| v6 fp16 + triton(修复) | 212.7s | 1.69e11 | 快 1.10× |
| v5 fp16 现网（对照） | 235.0s | 1.53e11 | — |
| v6 fp16 + triton（修复前，昨日） | 460.3s | 0.78e11 | 慢 0.96× |

（各数据集明细：noclamp 在 3t/glint/1201 上 30.4/72.9/12.5s，均为全矩阵最快）

### val yaml（3 个 custom_verification4，共 1.3e12 对）sim 合计

| 配置 | 合计耗时 | vs v5 |
|---|---|---|
| **v6 fp16 + torch + skip_clamp** | **6.6s** | **快 2.27×** |
| v6 tf32 + torch / shard | 8.2s | 快 1.83× |
| v6 fp16 + torch | 11.6s | 快 1.29× |
| v6 tf32 + triton(修复) | 11.2s | 快 1.34× |
| v6 fp16 + triton(修复) | 14.0s | 快 1.07× |
| v5 fp16 现网（对照） | 15.0s | — |

### 整轮耗时（含 TRT 特征提取与全部 evaluator）

| 轮次 | 耗时 | 内容 |
|---|---|---|
| test（44.7 min） | 特征提取 ~10.6 min + IJBC 两项 ~7 min + v6 矩阵 ~13 min + engine/其他 | 6 配置一次提特征循环复用（流程优化：省去 5×10.6 min 重复提取） |
| val（9.5 min） | 同结构 | |

### 优化尝试结论

1. **skip_clamp（fp16）**：省一趟 clamp 读写，提速 ~29-47%，pos 总数仍精确（fp16 归一化点积越界概率极低）。语义注意：越界值从"压入边界 bin"变为"丢弃"（与 v5 同语义）。已实现为 v6 的 `skip_clamp=True` 参数。
2. **shard 模式**：+5%（glint 96.7 vs 101.7s），大规模下收益更明显（生产 3342 万数据验证过）。
3. **流程优化**：`--v6_matrix` 一次提特征循环全部配置，本实验总时长从估算 4+ 小时压到 ~55 分钟。
4. triton 修复版在 bins=2000 下仍慢于 torch epi 25-30%（tl.histogram 路径 vs torch histc），但已从负优化变为可用（快于 v5）。

### 最终推荐（TRT 链路、bins=2000）

```
v6 + precision='fp16' + epilogue='torch' + skip_clamp=True + block=32768
```
比现网 v5 fp16 快约 2 倍，直方图精确。若不允许 skip_clamp 语义变化，用默认 clamp 版仍快 1.44×。

## 12. skip_clamp 语义修正：守恒补偿（9/13，响应评审问题）

问题指出：skip_clamp 原实现中越界值被 histc **直接丢弃**（不守恒），正确语义应为"越界算 ±1"（压入边界 bin）。

修复（cluster_utils_v6.py torch 路径）：skip_clamp 时统计每 tile 应有对数（n_valid/n_same，对角 tile 的 label_eq 对称计数需 //2 修正），histc 后将差额补入边界 bin。pos/neg 各自补偿。

验证：
1. 强越界场景（构造 sim>1 的 3,000 对）：skip_clamp+补偿 与 clamp 版**逐 bin 完全一致**，"越界算 1"语义等价 ✓
2. 20% 数据（666 万行）：noclamp 守恒补偿版 **73.0s（3.04e11 对/s）** vs clamp 版 97.8s —— **快 25%，pos 总数精确守恒（真值差 +0）**
3. bincount 路径（索引 clamp 天然守恒）结果一致

修正后推荐配置不变：`v6 + fp16 + torch epi + skip_clamp=True + blk=32768 + bins=2000`（73.0s@20% = 3.04e11 对/s，较 v5 fp16 现网快约 1.8×，语义无损）。

## 13. 补测：tf32 + bins=2000 组合（9/13，20% 排序数据，blk=32768）

| 配置 | 耗时 | 吞吐(对/s) | 真值差 | 逐 bin vs 首行(fp16/clamp) |
|---|---|---|---|---|
| **v6 tf32 + torch + skip_clamp** | **70.4s** | **3.15e11** | 0 | 否（精度差，预期） |
| v6 fp16 + torch + skip_clamp | 71.2s | 3.12e11 | 0 | —（基准） |
| v6 tf32 + torch | 95.5s | 2.33e11 | 0 | 否 |
| v6 fp16 + torch | 99.1s | 2.24e11 | 0 | — |
| v6 tf32 + triton(修复) | 126.7s | 1.75e11 | 0 | 否 |

结论：
1. **tf32 与 fp16 在 v6/bins=2k 下完全打平**（70.4 vs 71.2s；带 clamp 同样 tf32 略快）—— 确认 v6 瓶颈在 epilogue 不在 GEMM 精度。
2. **新全局最优：v6 + tf32 + torch epi + skip_clamp + blk=32768 + bins=2000 = 3.15e11 对/s**，比 fp16 noclamp 略快且数值更稳（无 fp16 半精度损失、无精度转换开销），比 v5 fp32 排序版快 3.3×，比 v5 fp16 现网参数快 1.9×。
3. skip_clamp 守恒补偿在 tf32 下同样工作正常（真值差 0）。
4. triton(修复) bins=2k 仍慢于 torch epi 约 32%。

## 14. 最终全链路实测：v6 最优配置 + TRT 提特征提速调研（9/13，test 全量）

最优 sim 配置：`v6 + tf32 + torch epi + skip_clamp + blk=32768 + bins=2000`。

### R1（bs256/workers5 现网提特征 + v6 最优 sim）vs 历史 v5 fp16

| 数据集 | v5 fp16 sim | v6 最优 sim | 提升 |
|---|---|---|---|
| test_3t (428万) | 65.3s | **31.5s** | 2.07× |
| test_enhance (20万) | 1.3s | **0.6s** | 2.2× |
| test_glint (682万) | 139.9s | **72.8s** | 1.92× |
| test_1201 (266万) | 28.5s | **12.8s** | 2.23× |
| **合计** | **235.0s** | **117.7s** | **2.0×** |

R1 整轮 29.5 分钟（v5 轮 33 分钟）。

### TRT 提特征提速调研：bs512/workers8 对照（R2）

| 数据集 | R1 bs256/w5 | R2 bs512/w8 | 差异 |
|---|---|---|---|
| test_3t | 285.7s | 323.5s | R2 慢 13% |
| enhance | 66.1s | 90.9s | R2 慢 37% |
| glint | 400.4s | 462.1s | R2 慢 15% |
| 1201 | 211.6s | 231.4s | R2 慢 9% |
| **合计** | **1015.7s** | **1154.3s** | **R2 慢 13.6%** |

结论：**提特征 batch 512 + workers 8 为负优化（慢 14%）**，现网 bs256/workers5 维持不变。sim 部分两轮复现一致（117.7 vs 118.7s）。提特征若需进一步提速，剩余方向为 int8 量化 engine（需校准）、图片预解码缓存等工程改造，简单调参已到头。

### 交付汇总

- 最终推荐：`v6 + tf32 + torch epi + skip_clamp + blk=32768 + bins=2000`（TRT 链路 `--sim_version v6` + 环境变量 `V6_PRECISION=tf32 V6_EPILOGUE=torch V6_SKIP_CLAMP=1`）
- 接入点：eval_all_trt_single.py 已支持（sim_version=v6 + 上述环境变量）；带样本对收集用 torch epi + collect_pairs_config（快 13%）
- v6 的 triton epilogue：轻分支（sorted neg 热循环）真加速已验证；重分支修复后可用作 tf32/200k bins 场景备选

## 15. 代码整合与后续优化（9/13）

### 15.1 代码整合（v6-only）

- `evaluations/cluster_utils_v6.py`（去 triton）已融合进 `evaluations/cluster_utils.py` 末尾，v6 文件已删除。triton kernel/TritonEpilogue 一并移除（仅保留说明性注释）。
- `get_sim_matrix_large_scale_v6` 保留函数名，签名去掉 `epilogue` 参数；`PairCollector` 复用 cluster_utils.py 现有实现，v6 的 `_collect_pairs` 以 `_collect_tile_pairs` 名字融入。
- 三个入口全面切 v6：`eval_all_trt_single.py`（删 --sim_version/--v6_precision/--v6_epilogue，固定 tf32+skip_clamp）、`custom_verification_evaluator.py`（type='4' 直接 v6，删 SIM_VERSION 环境变量）、`eval_all_torch_single.py`（走 evaluator 链路，无直接 v5 引用）。
- 说明：`cluster_utils.py` 中 v3/v4/v5 的历史函数定义保留（`eval_models_report.py`、`profile_v1_stages.py`、`custom_ijbbc_evaluator.py` 仍引用），三个指定入口已无 v5 调用。
- 双链路冒烟验证通过（TRT + Torch，TPIR 与历史一致）。

### 15.2 IJBC_gt_aligned 指标计算提速（225.4s → ~110s，2.1×）

瓶颈：`image2template_feature` 逐 template/media 的 Python 双重循环，且 3 种配置（Norm×Det）重复计算同一聚合结构。

修复：`evaluations/ijbbc/evaluate.py` 向量化重构（np.lexsort + np.add.reduceat 两段聚合），media 分组结构 3 配置共享只算一次；verification 批次 100000→500000。

对拍：与原实现逐元素一致（最大差 3.8e-8）。实测（真实 IJB-C 469,375 张 / 23,124 templates）：image2template ~60s→**4.0s（15×）**，单配置 36.4s，3 配置全程估计 **109s vs 225.4s**。

### 15.3 ijbc_custom（topk 路径，201.6s）—— 本轮未动

topk 协议（维护全局 top-k 负分）由 `get_sim_matrix_batch_balanced_silent`（v3 时代引擎）实现，sim 本体计算量 1.1e11 对是本质成本。后续可改造为 v6 引擎 + 直方图化 TPIR（协议语义需评审确认），预计 ~100s。

### 15.4 INT8 量化 engine —— 当前环境不可行

TRT 11.0 已移除 PTQ INT8 的简单接口（`BuilderFlag.INT8`/`FP16` 均不存在）；官方路径为 TensorRT Model Optimizer（modelopt，未安装）做显式 Q/DQ 量化。torch.ao 量化导出 QDQ ONNX 的 TRT 11 兼容性风险高。已清理本次的 int8 骨架代码；如需 int8，先安装 modelopt 并单独评估精度。

### 15.5 当前全链路耗时结构（test 轮）

| 环节 | 耗时 | 状态 |
|---|---|---|
| TRT 特征提取 | ~17 min | 简单调参已到头（bs512 负优化） |
| IJBC 两项指标 | ~7 min → **~4 min** | ijbbc 已 2.1×；ijbc_custom 待改 |
| sim 矩阵（4 cv4） | ~2 min | v6 最优配置（较 v5 快 2×） |
| 其他 | ~5 min | — |
| **整轮** | **29.5 → ~27 min** | 下一优先级：特征提取 int8/缓存、ijbc_custom 引擎替换 |

## 16. 特征提取提速专项：INT8 量化实测（9/13）

结论先行：**INT8 不可用（速度 0.51× + 精度崩至 80.6%）；engine 执行层已饱和无免费收益；提特征瓶颈在 CPU 解码（83% 时间），workers=5 已饱和。生产维持 fp16 + nw=5 不变。**

### 16.1 INT8 PTQ 链路打通（TRT 11 显式 Q/DQ）

15.4 节"不可行"结论修正：TRT 11 移除的是**隐式量化**接口（BuilderFlag.INT8 / IInt8Calibrator），**显式 Q/DQ 路径完整可用**，无需 modelopt。用环境内 onnxruntime-gpu 1.26 的 `quantize_static` 即可：

- 链路：fp32 PyTorch → 动态 batch fp32 ONNX → IJBC 真实对齐图 2048 张 MinMax 标定（per-channel 权重，QDQ 格式）→ QDQ ONNX → TRT engine（无需任何 flag）
- 踩坑 1：ORT 默认量化 bias 为 int32 Q/DQ 节点，TRT 只接受 8bit DQ → `extra_options={'QuantizeBias': False}`（bias 保持浮点，TRT 内部处理）
- 踩坑 2：ORT MinMax 默认非对称量化（zero_point≠0），TRT(GPU) 仅支持对称 → `extra_options={'ActivationSymmetric': True}`
- INT8 engine 构建成功（136s，72MB vs fp16 132MB，权重确实量化了）

### 16.2 INT8 结果：双重不可用

| 指标 | fp32 torch | fp16 TRT（生产） | int8 QDQ TRT |
|---|---|---|---|
| 纯 engine 吞吐（img/s, bs256） | — | **7,787** | 3,968（0.51×） |
| agedb_30 acc（10折，6000对） | 98.07±0.68 | 97.98±0.80 | **80.60±1.54** |
| 特征 cos vs fp32（mean / p1） | 1.0 | 0.99998 / 0.99996 | 0.576 / 0.076 |
| 端到端 1.2 万张（s） | — | 18.6 | 22.4 |

根因（build 日志证据）：ir101 的 **BN+PReLU 结构不支持 INT8**，每个 conv 前后都生成独立 Q/DQ 转换层（如 `Conv_output_0_DequantizeLinear`），int8 直连被 PReLU 边界切断。每 conv 的 int8→fp16→int8 requant 访存开销超过了 INT8 GEMM 的 2× 计算收益。精度崩坏源于对称量化下 ReLU 后激活只用一半动态范围 + PReLU 负半轴。**换无 PReLU 结构（如 ReLU 系 backbone）才有前提，但那需要重训模型。**

### 16.3 engine 执行层优化：全部无效（±1~2%）

fp16 engine 上实测（batch 256，输入常驻 GPU）：default stream（生产现状）7,808 img/s；专用非默认 stream 7,769；CUDA Graph 7,779；双 stream 双缓冲流水线 7,687。TRT 的 default stream 警告在本场景无实际影响——**engine 本体 GEMM 已饱和，launch/sync 开销可忽略，无免费收益。**

### 16.4 真正瓶颈：CPU 解码链路（83% 时间）

- 端到端（HF arrow → PIL decode → ToTensor → engine）：单卡稳态 **2,873 img/s**，仅为纯 engine 吞吐的 37%
- workers 扫描（IJBC 12 万张稳态，128 核 CPU）：nw=5 → 2,873；nw=10 → 2,659；nw=16 → 2,821；nw=24 → 2,586。**nw=5 已饱和，加 workers 反降**（瓶颈在内存带宽/解码实现，非并行度）
- 生产 7 卡 × nw=5 = 35 解码进程仅占 128 核 27%，但受限于单 worker 解码速率（~575 img/s）而非核数

### 16.5 提特征后续可挖方向（按预期收益排序）

1. **解码实现替换**：cv2.imread / pillow-simd / NVIDIA DALI（GPU 解码），预期 1.5~3×（DALI 可彻底消除 CPU 瓶颈，使端到端逼近 engine 吞吐 7,787）
2. **减少解码量**：评估图量削减/去重（协议允许范围内）
3. ~~INT8~~（16.2 已排除）、~~执行层优化~~（16.3 已排除）、~~engine batch/缓存~~（此前已排除）

### 16.6 测量方法教训（避免复现）

- flip 融合必须按 batch 粒度（engine 输出前半 normal 后半 flip），整表 `feats[:n]+feats[n:]` 会错位，导致 acc 假性随机（50.9%）
- benchmark 中 decode 必须放进 Dataset `__getitem__`（worker 进程），主进程 decode 会让 workers 数量实验完全失真（742 vs 2,533 img/s）
- fp32 torch acc=98.07% 自证了 agedb_30 的 issame 对齐与融合正确性，可作为精度对比的锚点



## 17. 解码加速专项研究（9/13）

结论先行：**数据是 PNG（非 JPEG），GPU 解码生态不可用；解码器替换 + pin_memory + nw=10 组合实测 3,464 img/s（vs 生产基线 ~2,900，+20%）；CPU 链路整体封顶 3,633（纯 engine 7,787 的 47%），提特征 17min → ~14min。**

### 17.1 单线程解码器对比（IJBC PNG 19.2KB/张，PNG 无损各解码器位一致 max_diff=0）

| 解码链 | img/s/线程 | vs 生产 |
|---|---|---|
| 生产（PIL + ToTensor + Normalize） | 1,054 | 1.00× |
| PIL 解码 + 手工 /255（免 transforms） | 1,477 | +40% |
| cv2.imdecode + cvtColor | 1,656 | +57% |
| torchvision decode_png | **1,718** | +63% |

生产链路 30% 开销在 torchvision transforms，不在解码本身。

### 17.2 端到端（IJBC 12 万张稳态，fp16 engine bs256）

| 配置 | img/s | 说明 |
|---|---|---|
| PIL 基线 nw=5（无 pin） | 2,610 | 生产同构 |
| cv2 nw=10 | 3,030 | +16% |
| **cv2 nw=10 + pin_memory（推荐）** | **3,464** | **+33% vs 无 pin 基线** |
| 零解码直通 + pin（上限） | 3,633 | 与推荐组合仅差 5% |
| 纯 engine 吞吐 | 7,787 | 理论上限 |

关键定位实验：**零解码直通也只有 3,108~3,633 img/s** → 瓶颈不在解码，在主进程串行链（collate → H2D → flip → engine → sync）。双缓冲双 stream（H2D 与 engine 重叠）零收益（3,633 vs 3,628）→ H2D 也不是瓶颈，剩余为 python per-batch 开销与 sync。pin_memory +11% 是主进程侧唯一有效优化。

### 17.3 GPU 解码路线：技术不可行

- 数据集存储为 **PNG**（nvJPEG 不适用）；`nvidia-nvimgcodec-cu12` 无 PNG GPU 后端（仅 nvjpeg/bmp/pnm）且无 python API 发布；DALI 2.1.0（pip nvidia-dali-cuda120）构建未注册 GPU 图像解码器（`decoders__Image not registered for gpu`）
- DALI CPU 解码 + external_source 喂入实测 1,541 img/s：单线程 python feed（逐样本拷贝，GIL）成为新瓶颈，反比基线慢
- arrow bytes 纯供给 21k img/s，供给从来不是瓶颈

### 17.4 生产落地建议（eval_all_trt_single.py）

1. `worker_extract*` 的 Dataset 改 `decode=False` 取 bytes + cv2.imdecode + 手工归一化（免 transforms），DataLoader 已是 pin_memory=True
2. `TRT_NUM_WORKERS` 默认 5 → 10
3. 预期收益：提特征 ~17min → ~14min（每卡 2,900 → 3,464 img/s），整轮 29.5 → ~24.5min
4. 若需突破 3,633 上限：改数据格式（PNG → raw uint8/JP×G，或转 MXNet rec + GPU 解码 JPEG），属数据工程而非解码器问题

环境变化：为评估安装了 `nvidia-dali-cuda120`（结论不可用，可卸载）与 `nvidia-nvimgcodec-cu12`（无用，可卸载），均未接入生产代码。

产物：`opt_eval/tensorrt/decode_bench.py`、`decode_e2e_bench.py`、`pipeline_bench.py`；结果 `decode_bench_results.parquet`、`decode_e2e_results.parquet`、`pipeline_results.parquet`。

### 17.5 落地实施与全量验证（9/13，已合入 eval_all_trt_single.py）

改动（4 处）：
1. `HFIndexedDataset`：`decode=False` 取压缩 bytes + cv2 解码 + 手工归一化（免 PIL/transforms），`with_path` 兼容 tinyface
2. 新增 `FastImageFolderDataset`：cv2 直读 ImageFolder 结构，label 按目录排序编号与 torchvision 一致；`worker_extract` 的 ImageFolder 分支换用（cv4 四集 1,397 万张 JPEG 走此路径）
3. `TRT_NUM_WORKERS` 默认 5 → 10（`DataLoader` 本就 pin_memory=True）
4. 主循环同轮特征复用：同一 HF 数据集（ijbbc/ijbc 共用 IJBC_gt_aligned）只提一次，第二次 `[feat-reuse]` 跳过；cv4 特征 ~57GB 不缓存

位一致性验证：HF 链路新旧输出 max_diff=0；FastImageFolderDataset 与 torchvision.ImageFolder 的样本数/path/label/顺序完全一致，解码差 <1e-6。

全量 test 轮实测（s4_0618，7 GPU，name=fastdecode_test_0913）：

| 数据集 | 提取耗时 | 吞吐 (forward/s/GPU) |
|---|---|---|
| work_0605_3t (428万) | 283.8s | 4,314 |
| work_0605_glint (682万) | 407.9s | 4,771 |
| work_1201 (267万) | 192.3s | 3,956 |
| work_0605_enhance (20万) | 62.6s | 927 (小集启动开销占比大) |
| IJBC_gt_aligned (47万, PNG) | 45.3s | 2,961 |
| ijbc (同数据集) | **0.0s (feat-reuse)** | — |

精度对拍（75 项 vs 历史轮）：平均差 0.027、最大 0.30（仅 FAR 1e-10/1e-6 极端端点）；IJBC PNG 链路位一致 + 特征复用段结果与历史几乎相同（001@1e-05: 96.2098 vs 96.2096）→ 差异为每轮重建 TRT engine 的固有 fp16 波动，与改动无关。

耗时：**整轮 29.5 → ~25.5min**（提特征段 ~21 → 16.5min，IJBC 少提一次省 ~45s）。

未动：Torch 链路（eval_all_torch_single.py / custom_verification_evaluator.py 内部加载）与本优化无关其结构，后续如需可同法改造；MXFaceDataset（rec 格式）分支保留原样（当前评估集均无 rec）。

### 17.6 "核多为什么不能更快"——供给链极限剖析（9/13）

问题：128 核 CPU、workers 加到 64，为何端到端仍 ~3,646 img/s（纯 engine 7,787）？

每批 256 张时间剖析（IJBC cv2 链，pin+nw=10）：

| 段 | 耗时 | 说明 |
|---|---|---|
| 等待下一批 (iter) | **35.0ms** | worker 解码+collate+shm 传输+pin |
| H2D | 0.1ms | pinned，几乎免费 |
| flip+cat | 0.1ms | GPU |
| engine+sync | **34.5ms** | 理论 33ms，**GPU 侧已满速** |

关键实验：
1. **纯供给速率**（只 iter 不跑 engine）：nw=10 → 7,343；nw=20 → 7,306；nw=40 → 8,550；nw=64 → 7,801。**与 worker 数无关** → DataLoader 批次交接链（worker 拼批 → shm → 队列 → pin 线程 → 主线程）存在与核数无关的单点，~7-8k img/s
2. **engine 环内 iter 等待 ≈ engine 耗时**（35 vs 34.5ms，1:1 停顿模式）→ 供给与 CUDA 同步交替耦合，有效吞吐减半至 ~3,646
3. **batch 512**（交接次数减半，重建 engine 实测）：3,458 vs 3,644，**无收益反降** → 不是每批固定开销，排除最后一条调参路线
4. pin=False 供给 +16%（8,516），仍远低于 16k 理论 → pin 也不是主限制

结论：**剩余瓶颈是单进程 python 数据管道架构成本（每张 ~0.13ms 的拼批/传输/包装 + 与 CUDA sync 的交替停顿），与核数无关**。workers/prefetch/batch 调参空间已穷尽。

突破只剩架构级方案（均为后续可选，本轮不实施）：
- **DALI `fn.readers.file` 直读 cv4 JPEG 目录**：全 C++ 链路绕开 python 喂入，理论逼近 7,400/卡（cv4 占提特征 95%，IJBC arrow 不适用但只占 5%）；接入成本：每 GPU worker 内嵌 pipeline 与 TRTInfer 衔接
- **每 GPU 2 个消费进程**分片喂同一 engine（双 context 排队）：预计 ~1.8×，中等工作量
- 接受现状：整轮已 29.5 → ~25.5min

### 17.7 架构方案 PoC 实测：三条路全部无法突破（9/13）

对 17.6 的两条架构路线 + 由此推出的第三条（线程分离），逐一 PoC 实测（IJBC 12 万张，fp16 engine bs256）：

| 方案 | 机制 | 实测 | vs 基线 3,644 |
|---|---|---|---|
| 双消费进程 / 卡 | 2 进程各持 context 分片喂同一 engine，停顿互相错开 | 2,818 img/s | **-23%** |
| DALI file_reader (CPU 解码 8/24 线程) | C++ 读文件+解码+归一化，绕开 python 喂入 | 3,288 / 3,297 img/s | **-10%** |
| 线程分离 | engine 独立线程执行 (sync 释放 GIL)，主线程专注收批重叠供给 | 3,516 img/s | **-3%** |

关键发现：
- 双进程负收益的机制：两进程需求叠加 (2×3,646=7,292) 恰好撞上纯供给墙 (~8k)，每进程分到的供给减半，叠加双 context 开销后反而更慢
- DALI 负收益的机制：其管线供给 (~5,900 img/s) 与解码线程数无关 (8→24 无变化)，加上同样的供给/计算交替停顿
- 线程分离无效的机制：纯供给 7,343 与 engine 上限 7,420 在理论上允许 ~7,300，但实际停在了单线程基线——供给端的 17.4ms/批 与 engine 的 34.5ms 无法有效重叠，指向 python/CUDA 驱动层更深的串行约束（GIL 调度粒度 / 默认流语义 / 驱动队列），非 PoC 级改动可解

**结论：~3,600 img/s/卡 是当前技术栈 (python DataLoader + TRT 默认流推理 + RTX 4090) 的实测天花板，5 条路径（workers 扫描 / 双缓冲 / 双进程 / DALI / 线程分离）交叉验证均无法突破。** 提特征段 16.5min 已达该架构的极限，整轮 25.5min 中剩余大头是指标计算 (~8min，ijbbc 已优化、ijbc_custom 201.6s 待 v6 引擎化) 与 sim 矩阵 (~2min，已最优)。

后续若需数量级提升，唯一明确可行的路径是 DALI GPU 解码（需 JPEG 数据格式 + nvjpeg 后端注册的 DALI 构建，当前 pip 版两者皆缺），属数据格式 + 环境构建工程，另立项评估。

## 18. ijbc_custom (topk 协议) 引擎替换：v6 直方图化（9/14，已合入）

### 18.1 协议语义与旧引擎瓶颈

`compute_metric_ijbc_custom` 维护全局 top-k（k=1e-3×总负对）负分定阈值 + 全量正对分数算 TPIR。旧引擎 `get_sim_matrix_batch_balanced_silent`（v3）：fp32 GEMM 本身仅 ~2-4s，89.8s 耗在每 tile 三次布尔掩码全量副本、负对全量 half().cpu() 传输、topk×2 粗筛堆合并。

### 18.2 探索过程（3 条路线）

| 路线 | 结果 |
|---|---|
| v6 直方图单遍 (tf32, 200k bins) | **2.2s（41×）**，far≥1e-8 与堆版一致（<0.02）；1e-10/1e-9 偏差 ~0.25 |
| 两遍窗口化（精细右尾直方图） | 修复 1e-8/1e-7，但 1e-10/1e-9 仍差 8×——负分密集区 bin 内多对无法分辨"第 N 大"，且 tf32 误差带（~1e-3）与 0.9995+ 右尾重叠 |
| 混合引擎（直方图+精确 topk 单遍，fp32） | 代码已写出并撤销（未接入）——按用户决策不走此路 |

结论：直方图分辨率对 far≤1e-9（top 11/110 个负对）天生不足；用户裁定这两个端点绝对值本就 <0.6，接受 ~0.25 偏差，**采用 v6 单遍方案**。

### 18.3 落地与端到端验证

`compute_metric_ijbc_custom` 的全量段与 001 子集段均替换为 v6（tf32 + skip_clamp + 200k bins）+ `compute_tpir_from_hist`；清理堆引擎相关 import。cluster_utils.py 的临时追加函数已删除恢复原状。

端到端实测（真实提特征 + 新函数，vs fork 轮旧引擎结果 18 项 TPIR）：
- **指标计算段 165.2s → 14.7s（11×）**；整轮 ijbc 条目 ~180s → ~30s
- 16/18 项差 <0.09；最大差 0.257 仅在 1e-10/1e-9 两个端点（用户已接受）；其余全部与历史一致

### 18.4 当前整轮耗时

| 环节 | 原始 | 当前 |
|---|---|---|
| TRT 特征提取 | ~21 min | 12.8 min |
| ijbc_custom 指标 | 165.2s | **14.7s** |
| IJBC 特征提取 | 提两次 | 提一次（feat-reuse） |
| **整轮** | **29.5 min** | **~22.5 min** |

### 18.5 训练期评估链路同步（9/14，已合入）

`train_opt.py` 训练期评估（默认 external_eval: False）走进程内 fabric evaluator 链路，其中 `CustomIJBCEvaluator`（custom_ijbbc_evaluator.py）是独立于 eval_all_trt_single 的另一份堆引擎实现。按用户决策**仅同步 v6 引擎**（cv2 解码 / fork 不动）：全量段与 001 子集段替换为 v6（tf32 + skip_clamp + 200k bins）+ `compute_tpir_from_hist`，输出 key 格式不变（all_/001_ 前缀），训练侧 summary 兼容。

端到端验证（真实提特征 + compute_metric，fabric=None 跳过 debug 检查）：compute_metric 13.4s；18 项 TPIR 与 eval_all_trt_single v6 版逐项一致（同引擎同特征），相对旧堆引擎结果仅 1e-10/1e-9 端点差 ~0.26（已裁定接受），无超 0.5 偏差项。训练期每个评估 epoch 的 ijbc_custom 指标段 ~165s → ~14s。


### 17.8 cv4 图片文件夹专项：spawn pickle 问题发现与 fork 修复（9/13，已合入）

用户指出 PoC 均在 IJBC 上做而 cv4（1,397 万张 JPEG，占提特征 95%）未单独调优——复查发现真实问题：

**问题**：`FastImageFolderDataset` 持有全量 (path, label) 列表，test_1201 达 **450MB pickle**（test_3t ~720MB、glint ~1.1GB）。DataLoader 默认 spawn 模式下，每个 GPU worker 进程要对 10 个 dataloader worker **串行 pickle/unpickle 10 次**——这批时间全部计入各集"特征提取"耗时且随集大小线性增长。

**实测**（test_1201 前 40 万张，单卡）：spawn 首批等待 **96.9s** vs fork **0.6s**，稳态吞吐两者相同（3,639 vs 3,617 img/s）——pickle 纯粹是启动税。

**修复**：三个 worker 的 DataLoader 指定 `multiprocessing_context='fork'`（dataloader worker 仅 CPU 解码，CUDA 已初始化但 fork 安全）；MXFaceDataset(rec) 分支保持默认 spawn（mxnet 对 fork 不安全，且当前无 rec 评估集）。

**全量 test 轮实测**（fork 版 vs 上轮）：

| 数据集 | spawn 轮 | fork 轮 | 节省 |
|---|---|---|---|
| work_0605_3t (428万) | 283.8s | 230.6s | -53s |
| work_0605_glint (682万) | 407.9s | 350.7s | -57s |
| work_1201 (267万) | 192.3s | 134.7s | -58s |
| work_0605_enhance (20万) | 62.6s | **21.9s** | -41s |
| IJBC_gt_aligned (47万) | 45.3s | 32.8s | -13s |
| ijbc (复用) | 0s | 0s | — |
| **提特征合计** | **991.9s** | **770.7s** | **-221s (-22%)** |

**整轮 25.5 → ~24min**。精度对拍：fork 轮 vs spawn 轮 75 项平均差 0.022（engine rebuild 固有波动），vs 原始基线平均差 0.019——fork 不改变任何数据与计算路径。

至此提特征段 21 → 12.8min（原始 spawn+PIL 链 → cv2+fork+nw10），叠加 IJBC 复用与 IJBC 指标优化，整轮从 29.5min 降至 ~24min。

产物：`opt_eval/tensorrt/int8_ptq_test.py`（全链路可复现）、`stream_pipeline_bench.py`、`dataloader_bench.py`；结果 `int8_ptq_results.parquet`、`stream_pipeline_results.parquet`、`dataloader_workers_results.parquet`。中间产物在 `/tmp/int8_ptq_work/`（含两个 engine，可删）。

# portable_eval · 可迁移评测工具包

双流 nvJPEG 读取（rec / folder）→ TRT 推理 → NxN 匹配 → TPIR@FPIR 的
一体化评测工具。**除 TRT engine 外零仓库依赖**，拷走本目录即可在其他环境
运行与调优。NxN 匹配内置生产 v7 同款融合核（源码随包分发，运行时编译、
失败自动回退纯 torch 实现）。

## 文件

| 文件 | 作用 |
|---|---|
| `check_env.py` | 环境自检：依赖/硬件/头文件/库逐项 ✓✗，缺什么给安装提示 |
| `make_synth_data.py` | 自造评测数据（ImageFolder 布局 + 同内容 rec 打包） |
| `eval_lib.py` | 工具库：依赖发现、nvjpeg 批量解码扩展、TRT engine、数据源、双流提取、NxN 匹配（融合核/torch）、TPIR |
| `run_eval.py` | 主入口：提特征(缓存) → 匹配 → TPIR 表 → JSON |
| `cuda_histpn_he_fused.cu` | 融合核源码（v7 同款：fp16 sim 直读 + full/pos 双桶直方图） |

## 依赖

必须：python≥3.10、torch(CUDA)、tensorrt、opencv-python、numpy、ninja、CUDA 头文件
nvjpeg 管线另需：`pip install nvidia-nvjpeg-cu12`（含头文件+库，自动发现；
也可 `export NVJPEG_HOME=...`）
兜底：无 nvjpeg 时用 `--pipeline cv2`（CPU 解码，慢但处处能跑）

## 迁移标准流程（三步）

```bash
# 1. 自检（迁移后首跑，最后两项会提示缺什么）
python check_env.py --compile-test --engine /path/engine --data /path/data

# 2. 自造数据全链路自检（20 档案 × 8 图，分钟级）
python make_synth_data.py --out /tmp/pe_data --sets "tiny:64:16"
python run_eval.py --engine /path/engine --data /tmp/pe_data/rec_tiny \
    --source rec --pipeline nvjpeg --gpus 1 --self-test

# 3. 真实数据评测
python run_eval.py --engine /path/engine \
    --data /data1/dataset_0918/test/test_34t --source folder \
    --pipeline nvjpeg --gpus 7 --tag mymodel_34t
```

## 自造数据的自检信号

同档案图 = 同一基底图 + 微噪声（近似重复图）。**硬信号是 `--self-test` 的
闭合校验**：提取数 == 扫描数、档案数一致、正对+负对 == N(N-1)/2。
本机 s4_0618 实测四种组合（folder/rec × nvjpeg/cv2）全部闭合，正对
64×C(16,2)=7,680 精确吻合，TPIR@1e-5 在 64~82%（数值取决于模型对噪声图
的输出分布，仅作参考不作判据；四种组合互差 ≤18% 属正常）。

实测发现并已内置规避：nvjpegDecodeBatched 对大批次总字节数有内部限制
（512×10.7KB 合成图会段错误，256×10.7KB 正常），故解码按 `--nvjpeg-chunk`
（默认 256）分片调用，TRT 仍按整批推理。真实人脸小图 (~3.3KB) 在 512 下
无此问题，但保守统一走 256 分片。

## 构建 engine（两种来源）

```bash
# A. 从 ONNX（纯 TRT API，无训练框架依赖，最适合迁移）
python run_eval.py --build-from-onnx model.onnx --engine-out my_fp16.engine --bs 512

# B. 从训练侧 model.pt（按 5.eval_trt.py 的 build_trt_engine_from_pt，需仓库代码）
```

## 主要调优参数（run_eval.py）

| 参数 | 作用 | 调优建议 |
|---|---|---|
| `--pipeline nvjpeg/cv2` | GPU 批量解码 / CPU 兜底 | 有 nvjpeg 库必选 nvjpeg |
| `--gpus` | 参与卡数 | 提特征与 NxN 匹配共用 |
| `--bs` | engine batch（静态构建时须与 ONNX 声明一致；动态构建时为 opt 点/缓冲容量） | 提取批大小 |
| `--dynamic-max` | >0 构建动态 batch engine（min=64/opt=--bs/max=该值） | 本模型实测无吞吐收益 |
| `--block` | NxN 匹配分块，0=按匹配器取默认（torch 8192 / fused 16384） | 越大越快越吃显存；OOM 减半 |
| `--matcher` | NxN 匹配实现：auto（默认，融合核优先失败回退 torch）/ fused / torch | 有 CUDA toolkit + ninja 时保持 auto |
| `--bins` | 直方图 bin 数 | 2000 为 fp16 推荐值 |
| `--decode-threads` | cv2 管线每卡解码线程 | nvjpeg 管线不适用 |

## 真实数据验证（s4_0618 engine，7×4090，test_enhance 203,497 张）

| 组合 | 聚合吞吐 | TPIR@1e-5 | TPIR@1e-7 |
|---|---:|---:|---:|
| folder + cv2（旧路线代理） | 31,442 | 64.10 | 0.191 |
| folder + nvjpeg 双流 | **46,269** | 63.45 | 0.187 |
| rec + nvjpeg 双流 | 43,572 | 63.45 | 0.187 |

- nvjpeg 双流比 cv2 路线快 **47%**（与 13 号专项脚本 +54% 一致），调优生效
- torch 版匹配器与生产 v7 交叉验证：TPIR@1e-5 63.45/64.10 vs v7 基线
  64.09（no-flip）/64.13（zarr）；@1e-7 0.187/0.191 vs 0.191 —— 实现正确
- 合成数据（10.7KB 噪声图）上同样复现排序：21,959 → 23,290 → 26,430

### 2026-09-20 二轮提速（匹配循环体 ~4.7×，数值逐位不变）

微基准（157,696 图 × 512 维，7×4090，直方图与原实现逐位一致）：

| 匹配实现 | tile 循环耗时 (157K) |
|---|---:|
| 原实现 | 1.23s |
| + bincount 去重（idx_p 原先算了 3 遍） | 0.26s |
| + 非常驻 A 块提升到 bj 循环外（H2D 流量↓） | 0.25s |
| + int32 索引（显存流量减半） | 0.22s |

已落进 `eval_lib.py`：匹配循环三处 + 后处理 fp16 直通
（归一化 fp32 逐块算直接落 fp16、缓存/匹配入口不再 fp32↔fp16 来回转换，
17.9M 规模省 ~1.5 min 与 ~52GB 峰值内存）；端到端 TPIR 与旧结果逐位一致。

**大规模实测（1.2M 合成特征，7 卡）——外推以此为准，勿用 157K 线性外推**
（157K 特征工作集 161MB 可驻 L2，吞吐被高估）：

| 匹配实现 | 1.2M 实测 | 17.9M 外推 |
|---|---:|---:|
| v7 融合核 `get_sim_matrix_large_scale_v7`（生产） | 3.7s（194 G 对/s） | ~14 min |
| torch 优化版（本工具包） | 18.3s（39.2 G 对/s） | ~68 min |

- 交叉点在 15 万~120 万张之间：157K 时 torch 版反而快 3.3×（v7 有 pin/注册等
  固定开销），≥百万级 v7 融合核稳定 5×
- v7 与 torch 版直方图不逐位一致：分块形状（16384 vs 8192）改变 cuBLAS 归约
  顺序，sim 末位 ulp 差使边界对挪 bin（~0.5% 对数挪 1 个 bin，bin 宽 1e-3），
  对 TPIR 影响可忽略；同分块同实现内仍是逐位确定

### 2026-09-20 三轮升级：生产 v7 融合核已内置（`--matcher auto` 默认走它）

把 v7 的 `cuda_histpn_he_fused.cu` 随包分发，`eval_lib.pos_neg_hist_fused`
按 v7 调用约定移植（torch GEMM fp16 → 单遍 kernel 直读 sim，shared-memory
full/pos 双桶原子聚合，免 fp32 物化与 int64 索引）；编译失败自动回退 torch 版。
157,696 图 × 512 维 7 卡实测（扩展编译缓存已热）：

| 匹配实现 | 157K 实测 | 与生产 v7 直方图 |
|---|---:|---|
| 移植融合核（本包 `pos_neg_hist_fused`） | **0.15s（84 G 对/s）** | **逐位一致** |
| 生产 v7（cluster_utils.get_sim_matrix_large_scale_v7） | 1.93s | — |
| torch 优化版（本包 `pos_neg_hist_torch`） | 0.60s | 边界挪 bin 0.00% |

- 比 v7 入口更快的原因：v7 每次调用对输入做 pin_memory 全量拷贝；移植版
  直接 cudaHostRegister 原缓冲（失败自动降级 pin_memory）
- 首次使用有一次性成本：nvcc 编译扩展约 80s（走 torch extensions 缓存，
  `check_env.py --compile-test` 可预编译），加上新进程首次加载/JIT 约 50s
  （进驱动缓存后消失；偶发缓存失效会再付一次加载开销，稳态匹配秒级）
- 端到端：7 卡 rec_perf 全链路 TPIR 与 torch 版逐位一致（62.753 @1e-5），
  匹配 2.3s（含进程级固定开销，稳态同上表）

提特征侧已到 TRT 推理上限（7,642 img/s/卡，worker 6,610）。**增大 batch 实测无收益**：
同一动态 batch engine 下 B=512/1024/2048 每张成本持平（7,326/7,405/7,470 img/s，
bs512 已饱和）——注意曾有"bs1024 快 2×"的假象，源于 `--bs` 不改变静态 ONNX
engine 的 batch、一半输出为垃圾值，被 TPIR 质量门识破。剩余提速选项：int8
engine（未测，需校准）。`--dynamic-max` 可构建动态 batch engine（TrtEngine 已支持），
本模型上不带来吞吐，仅为多 batch 复用提供便利。

`build_nvjpeg_ext` 现在会把 python 同级目录补进 PATH：用绝对路径调 python
（未 conda activate）时 pip 的 ninja 也能被找到，spawn worker 不再因此崩。
`build_engine_from_onnx` 已适配 TRT 11：弱精度 FP16 flag 被移除，
改用 STRONGLY_TYPED 网络，精度跟随 ONNX（fp16 导出即得 fp16 engine）。

### 2026-09-21 train_rec（38.8M 图）验证：nvjpeg+rec 生产可用

对 `/data1/dataset_0918/rec/train_rec`（38,796,753 图，train.rec 102GB 单文件）
实测（`15.bench_trainrec_nvjpeg.py`，每卡 250 批 × bs512 × 7 卡 = 896,000 图）：

| 管线 | 聚合 img/s | 备注 |
|---|---:|---|
| nvjpeg 双流 | 48,515 ~ 50,871 | 与 test rec 历史值 51.5K 一致 |
| cv2 兜底 | 36,149 | 同数据同卡 |

- **特征级影响**（`16.verify_trainrec_decode.py`，512 张逐张对比）：nvjpeg 与
  cv2 解码全像素平均差 1.35/255（分图最大差均值 11.2，视觉不可分），过
  s4_0618 TRT 后特征 cos mean 0.9989 / min 0.990 —— 与 TRT fp16 量化噪声
  同量级，业务可用；1e-9/1e-10 极档绝对值会有微小移动
- **新增 RecSource npy 缓存**：千万行级 train.idx/tsv 的 Python parse 每进程
  1-2 分钟，spawn 7 worker 原本各自重 parse；现首次 parse 落
  `train.offsets.npy` / `train.labels.npy`（tmp+原子替换防并发写坏），
  缓存命中打开 0.3-0.5s
- **新增 `max_batches` cfg 旋钮**：worker 级限量，bench/试跑用（全量提取
  仍用 extract_features）
- 自定义 probe 踩坑重申：`pe_decode_rec` 喂少于 init batch 的部分批同样段
  错误（见下节溯源），decode 批容量必须 == `pe_nvjpeg_init` 的 batch

## 已知口径与限制

- **nvjpeg 段错误溯源（2026-09-20，clean_enhance 管道排障结论）**：
  1. 炸点 = `nvjpegDecodeBatched` 用**少于初始化 batch 数的部分批**调用：
     内部状态损坏 → 句柄此后持续 INVALID_PARAMETER，或在宿主 libc memcpy
     段错误。与图片内容无关（出错子块逐张解码全部成功）、与输入缓冲生命
     周期无关（保活/pinned/sync=1 都不能根治）、与数据规模无关（永远发生
     在流尾唯一的非满子块）。本包 worker 从不发起部分批调用（尾部不足整批
     直接跳过，即 `n_skipped_tail`）——**该设计是 load-bearing 的**，自定义
     管线务必遵守：非满子块要么补满、要么走 cv2（示例见
     `6.数据集构建/clean_enhance/0.extract_enhance_feats.py` 的 `fill_sub`）。
  2. 加重因素 = 头/库版本错位：编译头文件来自 pip nvidia-nvjpeg-cu12
     (12.4.0.76)，运行时却可能按 soname 加载 ld.so.cache 里的系统库（本机
     `/usr/local/cuda-12.6` 的 12.3.3.54）。`build_nvjpeg_ext` 现已按
     `find_nvjpeg` 结果 ctypes 预载，保证头/库一致；旧库把「本该报错」
     处理成「直接段错误」，放大了排障难度。
- NxN 匹配三条路径口径一致（严格上三角每对计一次、bin 宽 1e-3）：融合核版
  与生产 v7 逐位一致；torch 版因分块形状不同存在边界对挪 bin（~0.5%，TPIR
  影响可忽略）。融合核版对 fp16 舍入致 |sim| 超出 [-1,1] 的对不计入
  （torch 版 clamp 进端点 bin），守恒校验允许 ≤0.001% 误差
- 融合核需要 CUDA toolkit + ninja 编译（一次性）；不可用时 `--matcher auto`
  自动回退 torch 版，`--matcher fused` 强制使用并在此前校验
- 提特征跳过尾部不足整批的图（run_eval 会打印跳过数量），闭合校验按保留数
- 双流管线的双缓冲/event 模式来自 13 号脚本，参数含义见其文件头注释

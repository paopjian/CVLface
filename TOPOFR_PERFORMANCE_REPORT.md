# TopoFR 训练慢速根因报告

统一测试条件：**8×GPU**(`MEMBER: 1/8`..`8/8` 已核实)、`batch_size=128`
**(每卡, 全局 batch=1024)**、bf16-mixed、IR-101、dataset_0605
(37,084,481 图 / **791,509 类**)、Lightning Fabric DDP、100 步计时。

**计时方法**：取每 10 步一次的 `Speed` 采样，去掉 tqdm 重复刷新的重复行，
丢弃第 1 次(warmup)，报**均值与区间**。不要用末值 —— test4 方差大(±9%)，
末值会系统性偏高。

## 1. 结论速览

| # | 代码 | 损失 | 分类头 | samples/s (均值) | 区间 | ms/步 | 状态 |
|---|------|------|--------|-----------------|------|-------|------|
| 1 | work_0605 | AdaFace | partial_fc | **5993** | 5795–6084 | 170.9 | 基线 |
| 2 | topofr | AdaFace | fc | **1880** | 1869–1888 | 544.7 | — |
| 3 | topofr | TopoFR | fc | **1662** | 1642–1688 | 616.1 | 用户观察到的"慢" |
| 4 | topofr | TopoFR | partial_fc | ~~崩溃~~ → **4914** | 4437–5309 | 208.4 | **修复后** |

慢速由**两个互相独立的原因**叠加，都不是 batch size、也不是数据增强：

1. **主因(3.19x)：用了 `fc` 而不是 `partial_fc`。** 791,509 类的全量 FC 层
   本身就把 5993 压到 1880，与 TopoFR 无关(测试 1 vs 2 都是 AdaFace)。
2. **次因(11.6%)：TopoFR 的持久同调损失。** 1880 → 1662。这部分是算法固有
   成本，量级正常，**不是**性能问题。

而测试 4(本该最快的组合)之所以一开始不可用，是 `topofr` 代码里的**两个真实
代码缺陷**——它们只在 `TopoFR loss + partial_fc` 同时启用时才暴露，所以之前
没被发现。修好后 TopoFR 达到基线的 **82%**，比测试 3 快 **2.96x**。

## 2. 先更正我之前的错误结论

我此前两版报告写过 "topofr 代码库比 work_0605 慢 3.2x，建议把 TopoFR 移植回
work_0605"。**这是错的**，用户的指正是对的：topofr 就是 work_0605 的复制。
我 diff 过两个目录树，除新增的 TopoFR pipeline/loss 外无实质差异。
6009 → 1879 的落差全部来自 **partial_fc → fc 的配置差异**，与代码库无关。
"移植"建议已作废。

## 3. 缺陷一：`-1` 标签撞上无 `ignore_index` 的 CrossEntropyLoss

**现象**：测试 4 在 8 卡上 8 个 rank 同时 SIGABRT，
`CUDA error: device-side assert triggered`。单卡跑同样配置**成功**(Loss 126.07)。
这个单卡/多卡分叉是定位的关键。

**根因**：PartialFC 是模型并行，每个 rank 只持有 `791,512/8 = 98,939` 个类中心，
非本地类的 label 会被掩成 **`-1`**(`partial_fc.py:176-177`)。而
`TopoFRLoss` 里为拿 per-sample loss 用了
`nn.CrossEntropyLoss(reduction="none")` —— 它**没有 `ignore_index`**，
`target=-1` 直接越界索引 → device-side assert。单卡时 world_size=1，
rank 持有全部类，不产生 `-1`，所以不崩。

**附带的正确性 bug**：即便不崩，只在本地 98,939 类上做 softmax 归一化在数学上
也是错的 —— 模型并行下 softmax 分母必须跨 rank 求和。

**修复**：新增 `DistCrossEntropyPerSampleFunc`
(`losses/topofr_loss.py`)，参照 PartialFC 自带的 `DistCrossEntropyFunc`
写成 **per-sample** 版本(原版只返回 mean，而 GUM 加权需要逐样本 loss)：

- `all_reduce(MAX)` 求全局 max_logits 做数值稳定
- `all_reduce(SUM)` 求全局 `sum_exp` 作为 softmax 分母
- 用 `index = where(label != -1)` 只在本地类上 gather 真值概率，再
  `all_reduce(SUM)` 拼回完整的 `prob_gt`(每个样本的真值必定落在且仅落在一个 rank)
- entropy 同样跨 rank `all_reduce(SUM)`，供 GUM 使用
- backward 用 `p_j - onehot_j`，`grad_loss` 已带下游 `.mean()` 的 `1/batch`

`TopoFRLoss.forward` 增加 `model_parallel` 开关，`fc` 路径保持原
`nn.CrossEntropyLoss` 不变，零回归。

**数值验证**(`test_dist_ce.py`，gloo/world_size=1/float64)：对
`F.cross_entropy` 与 `DistCrossEntropy` 逐项比对，前向、反向、加权梯度、
`label=-1` 路径共 6 项全部吻合到 **~1e-16**(机器精度)。

## 4. 缺陷二：all_gather 把拓扑损失的 N 从 128 撑到 1024

修掉崩溃后，测试 4 只跑出 **433 samples/s**(2365 ms/步) —— 比 `fc` 的 1657
还慢 4 倍，与我此前报告预测的 ~5300 完全相反。这说明还有第二个缺陷。

**根因**：PartialFC 为了类并行会 `all_gather` embeddings 和 images
(`partial_fc.py:167-170, 198-199`)，于是传给拓扑损失的 batch 变成
`128 × 8 = 1024`。而**持久同调是 CPU 上的超二次复杂度**算法
(Union-Find/MST，`topology.py` 里 `distances.detach().cpu().numpy()`)。

实测缩放(`test_topo_scaling.py`)：

| N | compute_topological_loss | 相对 N=128 |
|---|--------------------------|-----------|
| 128 | 29.6 ms | 1.0x |
| 256 | 114.5 ms | 3.9x |
| 512 | 512.2 ms | 17.3x |
| **1024** | **2288.5 ms** | **77.3x** |

即 N=1024 时拓扑单项 2288.5 ms，占 2365 ms 步时间的 **96.8%** —— 拓扑吃掉了
整个 step，PartialFC 省下的 FC 开销被完全盖过。

**修复**：拓扑损失是**逐 batch 的正则项**，没有跨 rank 语义需求；
`all_gather` 的存在只服务于类并行的 FC 层。改为把**本地** batch 传进去：

```python
input_images=input_images,      # 本地 batch, N=128
embeddings=local_embeddings,    # 本地 batch, N=128
model_parallel=True,
```

每个 rank 在自己的 128 上算拓扑，DDP 对梯度求平均 —— 与 `fc` 路径的语义一致。

**结果：4914 samples/s(均值)，0 崩溃。**

**交叉验证**：208.4 ms/步(测试 4) − 170.9 ms/步(测试 1 无拓扑基线) =
**37.5 ms**，与独立实测的 N=128 拓扑开销 **29.6 ms** 同量级(差值可由 GUM 的
CPU numpy 往返和 all_reduce 解释)。说明拓扑现在确实只跑在本地 batch 上 ——
若仍是 N=1024，这个差值会是 2288 ms 而非 37 ms。

## 5. 修复后的运行健康度

`test4_topofr_pfc_localtopo`，8 卡 / batch=128 / 100 步 / Epoch Time 1.53 min：

- 速度 **4914 samples/s** 均值，区间 4437–5309(±9%，每 10 步呈规律性起伏，
  疑为 dataloader 阶段性 stall，未深究)
- Loss **123.84 → 77.86** 单调下降(单卡对照首步 126.07，量级一致)
- 无 NaN/Inf(日志中 3 处 `nan|inf` 匹配均为 `INFO` 子串误命中)
- 8 个 rank 全部正常 `done` 退出，无 SIGABRT

## 6. 建议

1. **训练 TopoFR 请用 `classifiers=partial_fc/configs/...`**。791,509 类下
   全量 `fc` 是 3.2x 的纯浪费，这是本次调查中最大的单项收益。
2. 保持 `batch_size=128`(每卡)。鉴于拓扑的 77x 超二次曲线，**不要**为提速去加
   per-rank batch —— 加倍 batch 会让拓扑成本涨 ~3.9x，得不偿失。
3. TopoFR 剩余的 ~13% 开销是算法固有成本(CPU 持久同调)。若要继续压，
   方向是把 Union-Find/MST 搬上 GPU 或对 batch 做子采样，而不是调配置。

## 7. 待验证事项(诚实标注)

**逐 rank 拓扑 vs 全局 batch 拓扑对收敛/精度的影响尚未验证。** 本次改动让每个
rank 在自己的 128 样本上算持久同调，而非在 gather 后的 1024 上算。二者在数学上
不等价：拓扑结构是在更小的样本集上估计的。我的判断依据是拓扑损失作为逐 batch
正则项、且 `fc` 路径本来就是这个语义，但**这是设计推断，不是实验结论**。
需要一次完整训练跑比对验证精度是否受影响。若确认有损，替代方案是保留 gather
但对拓扑做子采样(如从 1024 中随机取 128)，成本相同而样本分布更广。


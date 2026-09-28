# TensorRT 评估精度排查与 FP16 加速报告

## 背景与问题

将评估推理从 PyTorch 切到 TensorRT 后, TRT 提取的特征与 PyTorch 严重不一致:
逐样本余弦相似度 `mean=0.70, min=-0.06` (正常应 >0.999)。但用同一份 ONNX 喂给
ONNX Runtime (ORT) 却完全正常 (cos≈1.0)。问题定位在 TRT 推理这一侧。

## 测试环境

- TensorRT: 11.0.0.114
- PyTorch: 2.12 (legacy TorchScript ONNX export, opset 17)
- 硬件: RTX 4090 (24GB)
- 模型: iResNet-101 (AdaFace/WebFace12M), 输入 112x112, 输出 512 维
- 测试集: 48091 张 / 1000 类 (ImageFolder)
- 指标: 逐样本 cos (vs Torch FP32) + acc + TPIR@FAR + 吞吐量

## 根因: IO binding 取反

推理器用 index 假设 IO tensor 顺序:

```python
self.input_name  = self.engine.get_tensor_name(0)   # 假设 0 = input
self.output_name = self.engine.get_tensor_name(1)   # 假设 1 = output
```

**TRT 10/11 不保证 IO tensor 的 index 顺序** —— ONNX 解析后 output 可能排到
index 0。一旦顺序与假设不符, 输入输出地址接反: 模型把输出 buffer 当输入读,
拿到的是上次结果 / 未初始化的 zeros, 因此 cos 崩到 0.7 甚至出现负值。

ORT 不受影响, 是因为它内部按 tensor 名字自动绑定 IO, 不依赖 index 顺序。

### 修复

按 `tensor_mode` 识别 input/output, 不依赖 index:

```python
import tensorrt as trt
self.input_name, self.output_name = None, None
for i in range(self.engine.num_io_tensors):
    n = self.engine.get_tensor_name(i)
    if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
        self.input_name = n
    else:
        self.output_name = n
```

修复后所有配置 cos 恢复到 0.9999+, TPIR/acc 与 Torch 无差异。

## 诊断方法 (二分定位)

为区分 "engine 算错" 还是 "提取流程错", 加了单 batch raw 诊断:

1. **raw_cos** (单 batch, 不经 flip/normalize): 判断 engine 本身是否算对
   - raw_cos≈1.0 但全量 cos 低 → 问题在 flip/拼接/normalize 流程
   - raw_cos 就低 → engine 计算错 (binding / 算子 / 构建参数)
2. **ORT 对照**: 同一 ONNX 跑 ORT, 验证导出的计算图正确 (排除 ONNX 导出问题)
3. **逐级 opt_level / 精度扫描**: 定位是否某个优化等级或精度引入误差

这套方法直接把问题锁定在 TRT binding, 而非 ONNX 导出或数值精度。

## 实测结果 (48091 张 / 1000 类)

| 配置 | cos_mean | cos_min | acc | TPIR@1e-6 | img/s |
|------|----------|---------|-----|-----------|-------|
| Torch FP32 (基准) | 1.0 | 1.0 | 99.9691 | 0.927472 | — |
| ONNX / ORT | 0.99999994 | 0.99999 | 99.9691 | 0.927463 | — |
| TRT opt3 FP32 | 0.99999994 | 0.99999654 | 99.9691 | 0.927475 | 981 |
| TRT opt5 FP32 | 0.99999994 | 0.99999595 | 99.9691 | 0.927475 | 978 |
| **TRT opt3 FP16** | 0.99997938 | 0.99963087 | 99.9690 | 0.927550 | **3414** |
| **TRT opt5 FP16** | 0.99998015 | 0.99985731 | 99.9692 | 0.927550 | **3462** |

注: img/s 为 7 评估器全量流水线中的单卡特征提取吞吐 (含 flip, dataloader)。
纯推理 benchmark (bs256) 下 opt3 FP16 约 7780 img/s。

## 关键结论

1. **binding 修复后 TRT 结果可信**: 全部配置 cos≈1.0, TPIR/acc 与 Torch 无差异。
2. **FP16 是最大加速杠杆**: 特征提取 981→3414 img/s (**3.5x**), 精度几乎无损
   (cos_min 0.99999→0.99963, TPIR@1e-6 反而略升, acc 持平)。4090 的 FP16
   Tensor Core 算力远超 FP32。
3. **opt_level 影响极小**: FP32 下 opt0~5 的 cos 全部 0.9999+; FP16 下 opt3 与
   opt5 推理速度接近 (3414 vs 3462), 但 opt5 build 慢得多。
4. **TF32 影响可忽略**: FP32 路径开/关 TF32 指标几乎一致。
5. **CUDA Graph 无收益**: FP16 + bs256 下是 GPU 计算瓶颈, kernel launch 占比极小,
   实测加速 +0.2% (噪声范围), 不值得增加复杂度。
6. **TRT 11 没有 `BuilderFlag.FP16`**: 计算精度由 ONNX dtype 决定。要 FP16 必须用
   `model.half()` 导出 FP16 ONNX, 而非 builder flag。

## build 速度 vs 推理速度 (FP16, 纯推理 benchmark)

| 配置 | build | 推理 |
|------|-------|------|
| opt0 FP16 | 6.6s | 6147 img/s |
| opt3 FP16 | 23.8s | 7780 img/s |
| opt5 FP16 | ~4min | ~7850 img/s |

opt_level 越高 build 越慢 (tactic 搜索空间更大), opt5 FP16 build 可达数分钟,
但相比 opt3 推理收益已饱和。评估场景 engine 只 build 一次、推理跑全量,
**opt3 是 build/推理的性价比拐点**。

## 推荐配置

**opt_level=3 + FP16** (`model.half()` 导出 FP16 ONNX): 特征提取 3414 img/s,
精度与 Torch 无差异, build 仅 24s。opt5 推理收益饱和但 build 慢约 10 倍, 不推荐。

## 落地

- binding 修正 (按 `tensor_mode` 识别 IO) 已同步到生产脚本
  `eval_all_trt_single.py` 的 `TRTInfer`。
- 排查 notebook: `opt_eval/tensorrt/test_trt_vs_torch_eval.ipynb`
  (含 Torch/ORT 基准 + 参数化 build + 逐级评估)。

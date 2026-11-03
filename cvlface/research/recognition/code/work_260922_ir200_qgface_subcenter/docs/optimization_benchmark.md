# train_opt.py 优化测速报告

## 测试环境

- PyTorch: 2.12.0+cu126
- 硬件: 7x RTX 4090 (24GB, PCIe)
- 模型: iResNet-101 (AdaFace/WebFace12M pretrained)
- 配置: batch_size=256, bf16-mixed, body.36 unfrozen (part_freeze)
- 数据: dataset_0605 (791509类, 37M图片)
- 测试方法: 每组跑 200 batch, 取后 100 batch 平均速度

## 优化项说明

1. **cudnn.benchmark**: `torch.backends.cudnn.benchmark = True`, 让 cuDNN 缓存最优卷积算法
2. **channels_last**: `model.to(memory_format=torch.channels_last)`, NHWC 内存格式加速卷积
3. **torch.compile**: `torch.compile(model, dynamic=False)`, 图编译优化 (默认 mode="default")

## 单项优化

| 优化 | 速度 (imgs/s) | vs baseline |
|------|--------------|-------------|
| baseline (train5.py) | 8,685 | — |
| cudnn.benchmark | 8,271 | -5% (无效) |
| channels_last | 9,907 | +14% |
| torch.compile | 9,238 | +6% |

## 两两组合

| 组合 | 速度 (imgs/s) | vs baseline |
|------|--------------|-------------|
| cudnn + channels_last | 10,384 | +20% |
| cudnn + compile | 11,561 | +33% |
| channels_last + compile | 11,157 | +28% |

## 三者全开 (train_opt.py 最终版)

| 组合 | 速度 (imgs/s) | vs baseline |
|------|--------------|-------------|
| cudnn + channels_last + compile | 11,381 | +31% |

## PyTorch 2.12 新特性测试

| 优化 | 速度 (imgs/s) | vs baseline | 结论 |
|------|--------------|-------------|------|
| mode="max-autotune-no-cudagraphs" | 8,710 (n=21) | 0% | 编译开销巨大(7min+), 稳定速度无提升, 不适合卷积层多的模型 |
| foreach=True (optimizer) | 10,948 | +26% | 无额外收益, Fabric内部已优化 |

## 尝试但不采用的优化

- **torch.compile mode="max-autotune-no-cudagraphs"**: 对 iResNet-101 (49个残差块) autotune 组合爆炸, 编译耗时>7分钟, 稳定速度反而无提升
- **optimizer foreach=True**: SGD 的 foreach 批量更新在 Fabric 环境下无额外收益 (10948 vs 11381)
- **optimizer fused=True**: 要求所有参数在同一设备, 与 partial_fc (跨GPU分片) 不兼容
- **model.compile() 方法**: PyTorch 2.12 推荐, 但 Fabric 的 setup() 流程中 torch.compile() 包装方式已经是先 compile 后 DDP, 效果一致

## 结论

- cudnn.benchmark 单独无效 (输入尺寸固定 112x112, cuDNN 默认已选最优算法)
- channels_last 单独最稳定 (+14%), NHWC 格式让 Tensor Core 更高效处理卷积
- torch.compile 单独 +6%, 但和 cudnn 组合后协同效果好 (+33%)
- 最佳组合: cudnn + compile (+33%), 三者全开 (+31%) 在误差范围内
- PyTorch 2.12 的新 compile mode 和 optimizer 优化对本场景无额外收益
- **train_opt.py 最终配置**: cudnn.benchmark + channels_last + torch.compile(mode="default")

## 使用方式

将 `流程.txt` 中阶段2-4的 `train5.py` 替换为 `train_opt.py` 即可。

注意: torch.compile 首次运行有 1-2 分钟编译开销, 之后每个 step 稳定加速。

## 日期

2026-06-09

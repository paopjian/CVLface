# TopoFR 训练脚本目录

## 📁 目录结构

```
scripts/train/
├── docs/                           # 所有文档
│   ├── QUICK_REFERENCE.txt        # 快速参考
│   ├── ALL_TRAINING_SCRIPTS.txt   # 所有脚本总览
│   ├── PRETRAINED_MODEL_GUIDE.md  # 预训练模型指南
│   ├── E2E_TRAINING_GUIDE.md      # 端到端训练指南
│   ├── TRAINING_SCRIPTS_GUIDE.txt # 脚本选择指南
│   ├── TOPOFR_README.md           # 完整使用手册
│   ├── TOPOFR_CHANGES.md          # 技术实现细节
│   ├── INTEGRATION_SUMMARY.md     # 集成总结
│   ├── CHECKLIST.md               # 检查清单
│   ├── PROJECT_COMPLETE.md        # 项目完成报告
│   └── README_TOPOFR.txt          # 简明README
│
├── 使用预训练模型 (推荐)
│   ├── train_topofr_from_pretrained.sh          ⭐⭐⭐ 标准fine-tune
│   ├── train_topofr_from_pretrained_quick.sh    快速验证
│   └── train_topofr_from_pretrained_aggressive.sh  激进策略
│
├── 从头训练
│   ├── train_topofr_e2e_full.sh                 标准训练
│   ├── train_topofr_e2e_quick.sh                快速验证
│   ├── train_topofr_e2e_cosine.sh               Cosine LR
│   ├── train_baseline_adaface_e2e.sh            Baseline对比
│   └── train_topofr_0605.sh                     多阶段训练
│
├── 辅助工具
│   ├── verify_files.sh            # 文件验证
│   └── test_topofr_integration.py # 集成测试
│
└── README.md                       # 本文件
```

## 🚀 快速开始

### 最推荐：使用预训练模型

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr/scripts/train

# 快速验证 (5 epochs, ~6h)
bash train_topofr_from_pretrained_quick.sh

# 完整训练 (15 epochs, ~20h) ⭐⭐⭐
bash train_topofr_from_pretrained.sh
```

## 📚 文档快速索引

- **快速开始**: `docs/QUICK_REFERENCE.txt`
- **所有脚本**: `docs/ALL_TRAINING_SCRIPTS.txt` 
- **预训练模型**: `docs/PRETRAINED_MODEL_GUIDE.md`
- **详细指南**: `docs/TOPOFR_README.md`

## 🎯 训练脚本选择

| 场景 | 脚本 | 时间 |
|------|------|------|
| 首次验证 | train_topofr_from_pretrained_quick.sh | ~6h |
| 生产训练 | train_topofr_from_pretrained.sh | ~20h ⭐ |
| Baseline对比 | train_baseline_adaface_e2e.sh | ~18h |

## 📞 获取帮助

查看 `docs/` 目录中的文档获取详细信息。

**预期性能**: 97%+ @ IJB-C (FAR=1e-5) 🎯

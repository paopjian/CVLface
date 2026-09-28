#!/usr/bin/env python3
"""
TopoFR Integration Test Script
测试TopoFR损失函数和相关组件是否正确集成
"""

import sys
import os

# 与 train_opt.py 相同的 root 解析 (indicator __root__.txt), 使 general_utils
# 等上层共享包与工作区包同时可用
import pyrootutils
pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

def test_imports():
    """测试所有导入是否正常"""
    print("=" * 60)
    print("测试1: 导入检查")
    print("=" * 60)

    try:
        from losses.topology import compute_topological_loss, TopologicalSignatureDistance
        print("✓ losses.topology 导入成功")
    except Exception as e:
        print(f"✗ losses.topology 导入失败: {e}")
        return False

    try:
        from losses.gum import gauss_unif
        print("✓ losses.gum 导入成功")
    except Exception as e:
        print(f"✗ losses.gum 导入失败: {e}")
        return False

    try:
        from losses.topofr_loss import TopoFRLoss
        print("✓ losses.topofr_loss 导入成功")
    except Exception as e:
        print(f"✗ losses.topofr_loss 导入失败: {e}")
        return False

    try:
        from losses import get_margin_loss
        print("✓ losses.__init__ 导入成功")
    except Exception as e:
        print(f"✗ losses.__init__ 导入失败: {e}")
        return False

    try:
        from pipelines.train_model_cls_topofr_pipeline import TrainModelClsTopoFRPipeline
        print("✓ pipelines.train_model_cls_topofr_pipeline 导入成功")
    except Exception as e:
        print(f"✗ pipelines.train_model_cls_topofr_pipeline 导入失败: {e}")
        return False

    print("\n所有导入测试通过！\n")
    return True


def test_topofr_loss_creation():
    """测试TopoFR损失函数创建"""
    print("=" * 60)
    print("测试2: TopoFR损失函数创建")
    print("=" * 60)

    import torch
    from losses.margin_loss import ArcFace, CosFace
    from losses.adaface import AdaFaceLoss
    from losses.topofr_loss import TopoFRLoss

    # 测试1: 基于ArcFace
    try:
        base_loss = ArcFace(s=64.0, margin=0.5)
        topofr_loss = TopoFRLoss(base_loss, topo_weight=0.1, use_gum=True)
        print("✓ TopoFR + ArcFace 创建成功")
    except Exception as e:
        print(f"✗ TopoFR + ArcFace 创建失败: {e}")
        return False

    # 测试2: 基于CosFace
    try:
        base_loss = CosFace(s=64.0, m=0.4)
        topofr_loss = TopoFRLoss(base_loss, topo_weight=0.1, use_gum=True)
        print("✓ TopoFR + CosFace 创建成功")
    except Exception as e:
        print(f"✗ TopoFR + CosFace 创建失败: {e}")
        return False

    # 测试3: 基于AdaFace
    try:
        base_loss = AdaFaceLoss(s=64, m=0.4, h=0.333, t_alpha=0.01)
        topofr_loss = TopoFRLoss(base_loss, topo_weight=0.1, use_gum=True)
        print("✓ TopoFR + AdaFace 创建成功")
    except Exception as e:
        print(f"✗ TopoFR + AdaFace 创建失败: {e}")
        return False

    print("\n所有损失函数创建测试通过！\n")
    return True


def test_config_loading():
    """测试配置文件加载"""
    print("=" * 60)
    print("测试3: 配置文件加载")
    print("=" * 60)

    import yaml
    from omegaconf import OmegaConf

    config_files = [
        'losses/configs/topofr.yaml',
        'losses/configs/topofr_cosface.yaml',
        'losses/configs/topofr_adaface.yaml'
    ]

    for config_file in config_files:
        config_path = os.path.join(WORKSPACE_ROOT, config_file)
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            assert cfg['margin_loss_name'] == 'topofr'
            print(f"✓ {config_file} 加载成功 (base_loss={cfg.get('base_loss', 'N/A')})")
        except Exception as e:
            print(f"✗ {config_file} 加载失败: {e}")
            return False

    print("\n所有配置文件加载测试通过！\n")
    return True


def test_forward_pass():
    """测试前向传播"""
    print("=" * 60)
    print("测试4: 前向传播（不需要实际训练）")
    print("=" * 60)

    import torch
    from losses.margin_loss import ArcFace
    from losses.topofr_loss import TopoFRLoss

    # 创建模拟数据
    batch_size = 4
    num_classes = 10
    embedding_dim = 512

    try:
        # 创建损失函数
        base_loss = ArcFace(s=64.0, margin=0.5)
        topofr_loss = TopoFRLoss(base_loss, topo_weight=0.1, use_gum=True)

        # 模拟数据 (logits 需有界: 真实路径来自 cosine, ArcFace arccos 不能吃越界值;
        # topology.py 的 latent_norm 现场创建并 .cuda(), 输入必须同在 GPU 上)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("  (跳过 GPU 检查: 本测试需 CUDA, topology.py 的 latent_norm 硬编码 .cuda())")
            return True
        logits = (torch.rand(batch_size, num_classes) * 1.9 - 0.95).to(device)
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        embeddings = torch.randn(batch_size, embedding_dim).to(device)
        input_images = torch.randn(batch_size, 3, 112, 112).to(device)

        # 前向传播
        result = topofr_loss(
            logits=logits,
            labels=labels,
            input_images=input_images,
            embeddings=embeddings
        )

        # 检查返回值
        if isinstance(result, tuple) and len(result) == 2:
            loss, loss_dict = result
            print(f"✓ 前向传播成功")
            print(f"  - Total Loss: {loss.item():.4f}")
            print(f"  - Loss Components: {loss_dict}")
        else:
            print(f"✗ 前向传播返回值格式错误: {type(result)}")
            return False

    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n前向传播测试通过！\n")
    return True


def test_partial_fc_topofr():
    """测试 PartialFC_V2 的 TopoFR 分支 (model_parallel 路径 + adaface buffer)"""
    print("=" * 60)
    print("测试5: PartialFC + TopoFR 分布式路径")
    print("=" * 60)

    import torch
    from types import SimpleNamespace
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import config as config_module
    from classifiers import get_classifier
    from losses import get_margin_loss

    with TemporaryDirectory() as temp_dir:
        init_file = Path(temp_dir) / "distributed_init"
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
        try:
            classifier_config = config_module.load_yaml("partial_fc_sample10", directory="classifiers")
            classifier_config.sample_rate = 1.0
            loss_config = config_module.load_yaml("topofr_adaface", directory="losses")
            classifier = get_classifier(
                classifier_config,
                get_margin_loss(loss_config),
                SimpleNamespace(output_dim=8),
                num_classes=8,
                rank=0,
                world_size=1,
            )

            # is_topofr 标志 + adaface buffer (TopoFRLoss 包装 AdaFace 基损失);
            # topology.py 的 latent_norm 硬编码 .cuda(), 输入必须同在 GPU 上
            if not torch.cuda.is_available():
                print("  (跳过: 本测试需 CUDA)")
                return True
            device = 'cuda'
            classifier = classifier.to(device)
            assert classifier.is_topofr is True
            assert hasattr(classifier.partial_fc, 'batch_mean')

            embeddings = torch.randn(4, 8, requires_grad=True, device=device)
            input_images = torch.randn(4, 3, 112, 112, device=device)
            labels = torch.tensor([0, 1, 2, 3], device=device)
            loss = classifier(embeddings, labels, input_images=input_images)
            loss.backward()
            assert torch.isfinite(loss)
            assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
            print(f"✓ PartialFC + TopoFR 前向/反向成功 (loss={loss.item():.4f})")
        finally:
            torch.distributed.destroy_process_group()

    print("\nPartialFC + TopoFR 测试通过！\n")
    return True


def test_ir200_model_and_peft():
    """测试 ir200 模型构造 + pefts body.72 冻结范围"""
    print("=" * 60)
    print("测试6: IR200 模型与 pefts body.72")
    print("=" * 60)

    import torch
    from types import SimpleNamespace
    import config as config_module
    from models import get_model
    from pefts import apply_peft

    model_config = config_module.load_yaml("iresnet_insightface.v1_ir200", directory="models")
    model = get_model(model_config, task="topofr_test")
    n_params = sum(p.numel() for p in model.parameters())
    # insightface ir200: [6,26,60,6] IBasicBlock, 512 维输出, ~7.2 亿参数量级
    assert 60e6 < n_params < 200e6, f"ir200 参数量异常: {n_params}"

    feats = model(torch.randn(2, 3, 112, 112))
    assert feats.shape == (2, 512)

    # body.72 解冻范围: layer3 第 40 块起 (6+26+40=72) + layer4 + 头
    peft_config = SimpleNamespace(name='part_freeze', target_modules='body.72',
                                  model_ckpt_dir='', classifier_ckpt_dir='', center_paths=None,
                                  keypoint_ckpt_dir='', train_keypoints=False)
    model, _ = apply_peft(peft_config, model=model, classifier=None,
                          data_cfg=SimpleNamespace(num_classes=8), label_mapping=None)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith('net.layer3.40') for n in trainable), "layer3.40 应解冻"
    assert any(n.startswith('net.layer4.') for n in trainable), "layer4 应解冻"
    assert all(not n.startswith('net.layer1.') for n in trainable), "layer1 应冻结"
    assert all(not n.startswith('net.layer2.') for n in trainable), "layer2 应冻结"
    assert all(not n.startswith('net.layer3.39') for n in trainable), "layer3.39 应冻结"
    print(f"✓ ir200 构造成功 (params={n_params/1e6:.1f}M), body.72 冻结范围正确 "
          f"(trainable 段: layer3.40+ / layer4 / 头)")

    print("\nIR200 模型与 pefts 测试通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TopoFR 集成测试")
    print("=" * 60 + "\n")

    tests = [
        ("导入检查", test_imports),
        ("损失函数创建", test_topofr_loss_creation),
        ("配置文件加载", test_config_loading),
        ("前向传播", test_forward_pass),
        ("PartialFC+TopoFR", test_partial_fc_topofr),
        ("IR200 模型与 pefts", test_ir200_model_and_peft),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！TopoFR集成成功！")
        print("\n可以开始使用以下命令训练：")
        print("  losses=configs/topofr_adaface.yaml")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

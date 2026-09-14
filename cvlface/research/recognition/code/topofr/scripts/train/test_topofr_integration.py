#!/usr/bin/env python3
"""
TopoFR Integration Test Script
测试TopoFR损失函数和相关组件是否正确集成
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr')

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
        config_path = os.path.join('/root/zhaokj/CVLface/cvlface/research/recognition/code/topofr', config_file)
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

        # 模拟数据
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        embeddings = torch.randn(batch_size, embedding_dim)
        input_images = torch.randn(batch_size, 3, 112, 112)

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

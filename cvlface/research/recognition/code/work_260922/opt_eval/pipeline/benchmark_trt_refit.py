"""
TensorRT Refit 验证脚本
验证: 构建一次 REFIT engine，之后只替换权重 (~1s) 而不用重新编译 (~56s)

核心: ONNX 导出会将 BN 折叠进 Conv，所以 refit 时必须提供融合后的权重。
方案: PyTorch 侧先 fuse_bn → 导出融合模型 ONNX → refit 时同样 fuse_bn 后映射。

流程:
1. 融合 BN 到 Conv（PyTorch 侧）
2. 导出融合后模型的 FP16 ONNX
3. 构建 REFIT-enabled engine (一次性 ~25s)
4. 用不同 checkpoint 的融合权重 refit engine (~0.1s)
5. 正确性验证

用法:
  python benchmark_trt_refit.py
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os, sys, time, copy
sys.path.append(os.path.join(root))

import numpy as np
np.bool = np.bool_

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_model
from general_utils.config_utils import load_config


CACHE_DIR = '/tmp/trt_refit_cache'
ONNX_PATH = os.path.join(CACHE_DIR, 'iresnet101_fp16_fused.onnx')
ENGINE_PATH = os.path.join(CACHE_DIR, 'iresnet101_fp16_fused_refit.engine')
BATCH_SIZE = 256


def load_model(ckpt_path):
    """加载 PyTorch 模型"""
    model_config = load_config(os.path.join(ckpt_path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(ckpt_path, 'model.pt'))
    model.eval()
    return model


def fuse_bn_into_conv(model):
    """
    将模型中的 Conv+BN 融合为单个 Conv (原地修改)。
    融合后 BN 变为 Identity，权重直接在 Conv 中。
    使用 PyTorch 内置的 fuse_modules 或手动融合。
    """
    fused_model = copy.deepcopy(model)
    fused_model.eval()
    # 使用 torch.ao.quantization 的融合工具
    fused_model = torch.ao.quantization.fuse_modules_qat(
        fused_model, [], inplace=False
    )
    # 上面的 fuse 需要指定具体模块路径，对复杂模型不方便
    # 改用 torch.fx 自动融合
    try:
        from torch.fx.experimental.optimization import fuse
        fused_model = fuse(fused_model)
        return fused_model
    except Exception:
        pass

    # 手动融合: 遍历所有 (Conv, BN) 相邻对
    return _manual_fuse_bn(copy.deepcopy(model))


def _fuse_conv_bn(conv, bn):
    """将 Conv 后面的 BN 融合进 Conv 权重 (Conv→BN)"""
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_gamma = bn.weight
    bn_beta = bn.bias
    eps = bn.eps

    scale = bn_gamma / torch.sqrt(bn_var + eps)

    # conv weight shape: [out_channels, in_channels, kH, kW]
    w = conv.weight.data
    w_fused = w * scale.view(-1, 1, 1, 1)

    if conv.bias is not None:
        b = conv.bias.data
    else:
        b = torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype)
    b_fused = (b - bn_mean) * scale + bn_beta

    return w_fused, b_fused


def _fuse_bn_conv(bn, conv):
    """将 Conv 前面的 BN 融合进 Conv 权重 (BN→Conv)
    BN: y = scale * x + offset, where scale=gamma/sqrt(var+eps), offset=beta-scale*mean
    Conv(BN(x)): W_new[o,c,kh,kw] = W[o,c,kh,kw] * scale[c]
                 b_new[o] = b[o] + sum_c(offset[c] * sum_kh_kw(W[o,c,kh,kw]))
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    offset = bn.bias - scale * bn.running_mean

    w = conv.weight.data
    # scale per input channel
    w_fused = w * scale.view(1, -1, 1, 1)

    if conv.bias is not None:
        b = conv.bias.data
    else:
        b = torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype)
    # W shape: [out_c, in_c, kH, kW] → sum spatial → [out_c, in_c] @ offset[in_c] → [out_c]
    b_fused = b + (w.sum(dim=[2, 3]) @ offset)

    return w_fused, b_fused


def _manual_fuse_bn(model):
    """手动遍历模型，融合所有 Conv+BN 和 BN+Conv 对"""
    fuse_count_post = 0  # Conv→BN
    fuse_count_pre = 0   # BN→Conv

    def fuse_children(module):
        nonlocal fuse_count_post, fuse_count_pre
        children = list(module.named_children())
        for i in range(len(children) - 1):
            name1, child1 = children[i]
            name2, child2 = children[i + 1]
            # Conv→BN
            if isinstance(child1, nn.Conv2d) and isinstance(child2, nn.BatchNorm2d):
                w_fused, b_fused = _fuse_conv_bn(child1, child2)
                child1.weight.data = w_fused
                if child1.bias is None:
                    child1.bias = nn.Parameter(b_fused)
                else:
                    child1.bias.data = b_fused
                setattr(module, name2, nn.Identity())
                fuse_count_post += 1
            # BN→Conv
            elif isinstance(child1, nn.BatchNorm2d) and isinstance(child2, nn.Conv2d):
                w_fused, b_fused = _fuse_bn_conv(child1, child2)
                child2.weight.data = w_fused
                if child2.bias is None:
                    child2.bias = nn.Parameter(b_fused)
                else:
                    child2.bias.data = b_fused
                setattr(module, name1, nn.Identity())
                fuse_count_pre += 1

        # 递归处理子模块
        for name, child in module.named_children():
            if not isinstance(child, nn.Identity):
                fuse_children(child)

    fuse_children(model)
    print(f"  手动融合: {fuse_count_post} 个 Conv→BN, {fuse_count_pre} 个 BN→Conv")
    return model


def export_fused_onnx_fp16(model):
    """导出融合后模型的 FP16 ONNX"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_fp16 = model.half().cuda()
    dummy = torch.randn(BATCH_SIZE, 3, 112, 112, device='cuda', dtype=torch.float16)

    print(f"  导出 ONNX: {ONNX_PATH}")
    with torch.no_grad():
        torch.onnx.export(
            model_fp16, dummy, ONNX_PATH,
            input_names=['input'], output_names=['output'],
            dynamic_axes=None, opset_version=17, dynamo=False,
        )
    size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"  ONNX 大小: {size_mb:.1f} MB")

    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    n_inits = len(onnx_model.graph.initializer)
    print(f"  ONNX 权重数: {n_inits}")
    return n_inits


def build_refit_engine(unused=None):
    """构建启用 REFIT 的 TensorRT engine"""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse Error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # 启用 REFIT
    config.set_flag(trt.BuilderFlag.REFIT)
    print("  REFIT flag 已启用")

    print("  构建 engine (首次, 含优化)...")
    build_start = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_time = time.time() - build_start

    if serialized is None:
        print("  Engine 构建失败!")
        return None

    print(f"  Engine 构建耗时: {build_time:.1f}s")
    print(f"  Engine 大小: {serialized.nbytes/1024/1024:.1f} MB")

    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized)
    print(f"  保存到: {ENGINE_PATH}")

    return build_time


def get_onnx_weight_map(model):
    """
    构建 PyTorch state_dict key -> ONNX initializer name 的映射。
    利用 ONNX 图的拓扑结构：
    - Conv 节点 inputs: [data, weight, bias]
    - PRelu 节点 inputs: [data, slope]
    按拓扑序遍历，和 PyTorch named_modules 的顺序对应。
    """
    import onnx
    from onnx import numpy_helper
    onnx_model = onnx.load(ONNX_PATH)

    # 收集所有 initializer 名称集合
    init_names = {init.name for init in onnx_model.graph.initializer}

    # 按图的拓扑序，提取每个算子引用的 initializer
    # Conv: input[1]=weight, input[2]=bias (可选)
    # PRelu: input[1]=slope
    # Gemm (FC): input[1]=weight, input[2]=bias
    # BatchNormalization: input[1]=scale, input[2]=bias, input[3]=mean, input[4]=var
    onnx_param_order = []  # [(onnx_init_name, role), ...]

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            if len(node.input) > 1 and node.input[1] in init_names:
                onnx_param_order.append((node.input[1], 'conv_weight'))
            if len(node.input) > 2 and node.input[2] in init_names:
                onnx_param_order.append((node.input[2], 'conv_bias'))
        elif node.op_type == 'PRelu':
            if len(node.input) > 1 and node.input[1] in init_names:
                onnx_param_order.append((node.input[1], 'prelu_slope'))
        elif node.op_type == 'Gemm':
            if len(node.input) > 1 and node.input[1] in init_names:
                onnx_param_order.append((node.input[1], 'fc_weight'))
            if len(node.input) > 2 and node.input[2] in init_names:
                onnx_param_order.append((node.input[2], 'fc_bias'))
        elif node.op_type == 'BatchNormalization':
            # 融合后不应该有 BN 了，但以防万一
            if len(node.input) > 1 and node.input[1] in init_names:
                onnx_param_order.append((node.input[1], 'bn_scale'))
            if len(node.input) > 2 and node.input[2] in init_names:
                onnx_param_order.append((node.input[2], 'bn_bias'))
            if len(node.input) > 3 and node.input[3] in init_names:
                onnx_param_order.append((node.input[3], 'bn_mean'))
            if len(node.input) > 4 and node.input[4] in init_names:
                onnx_param_order.append((node.input[4], 'bn_var'))

    print(f"  ONNX 图中按拓扑序提取到 {len(onnx_param_order)} 个参数引用")

    # PyTorch 侧: 按 named_modules 顺序提取参数
    # 对融合模型: Conv 有 weight+bias, PReLU 有 weight, Identity 无参数
    pt_param_order = []  # [(state_dict_key, role), ...]

    state_dict = model.half().state_dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            key_w = name + '.weight'
            if key_w in state_dict:
                pt_param_order.append((key_w, 'conv_weight'))
            key_b = name + '.bias'
            if key_b in state_dict:
                pt_param_order.append((key_b, 'conv_bias'))
        elif isinstance(module, nn.PReLU):
            key_w = name + '.weight'
            if key_w in state_dict:
                pt_param_order.append((key_w, 'prelu_slope'))
        elif isinstance(module, nn.Linear):
            key_w = name + '.weight'
            if key_w in state_dict:
                pt_param_order.append((key_w, 'fc_weight'))
            key_b = name + '.bias'
            if key_b in state_dict:
                pt_param_order.append((key_b, 'fc_bias'))
        elif isinstance(module, nn.BatchNorm2d) and not isinstance(module, nn.Identity):
            # 融合后的残余 BN (如果有)
            for suffix, role in [('.weight', 'bn_scale'), ('.bias', 'bn_bias'),
                                 ('.running_mean', 'bn_mean'), ('.running_var', 'bn_var')]:
                key = name + suffix
                if key in state_dict:
                    pt_param_order.append((key, role))

    print(f"  PyTorch 模型按模块序提取到 {len(pt_param_order)} 个参数")

    # 按 role 分组，然后按序对应
    from collections import defaultdict
    onnx_by_role = defaultdict(list)
    pt_by_role = defaultdict(list)

    for oname, role in onnx_param_order:
        onnx_by_role[role].append(oname)
    for pkey, role in pt_param_order:
        pt_by_role[role].append(pkey)

    mapping = {}
    for role in onnx_by_role:
        onnx_list = onnx_by_role[role]
        pt_list = pt_by_role.get(role, [])
        n_match = min(len(onnx_list), len(pt_list))
        if len(onnx_list) != len(pt_list):
            print(f"  警告: role={role} ONNX有{len(onnx_list)}个, PyTorch有{len(pt_list)}个, 取min={n_match}")
        for i in range(n_match):
            mapping[pt_list[i]] = onnx_list[i]

    print(f"  最终映射: {len(mapping)}/{len(state_dict)} PyTorch keys → ONNX names")

    # 统计
    unmatched_pt = [k for k in state_dict.keys()
                    if k not in mapping and 'num_batches_tracked' not in k]
    if unmatched_pt:
        print(f"  未匹配 (非 BN tracked, 前5): {unmatched_pt[:5]}")

    return mapping


def refit_engine_with_weights(state_dict, weight_mapping):
    """用新权重 refit engine，返回耗时"""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(ENGINE_PATH, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(memoryview(f.read()))

    refitter = trt.Refitter(engine, logger)

    # 获取 engine 中所有可 refit 的权重名
    refittable = set(refitter.get_all_weights())
    print(f"  Engine 可 refit 权重数: {len(refittable)}")

    refit_start = time.time()
    refit_count = 0

    for pt_key, onnx_name in weight_mapping.items():
        if onnx_name not in refittable:
            continue
        weight = state_dict[pt_key]
        weight_np = weight.cpu().numpy()
        # TRT 11: 使用 set_named_weights (name, trt.Weights)
        refitter.set_named_weights(onnx_name, trt.Weights(weight_np))
        refit_count += 1

    # 执行 refit
    success = refitter.refit_cuda_engine()
    refit_time = time.time() - refit_start

    if not success:
        print(f"  Refit 失败!")
        missing = refitter.get_missing_weights()
        if missing:
            print(f"  缺失权重 ({len(missing)}): {missing[:5]}...")
        return None, refit_time

    print(f"  Refit 成功: {refit_count} 个权重, 耗时 {refit_time:.2f}s")
    return engine, refit_time


def run_trt_inference(engine, input_tensor):
    """TRT 推理 (固定 batch size, 不足时 padding)"""
    import tensorrt as trt

    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    actual_bs = input_tensor.shape[0]
    # 必须 padding 到 BATCH_SIZE (engine 固定 batch)
    if actual_bs < BATCH_SIZE:
        pad = torch.zeros(BATCH_SIZE - actual_bs, 3, 112, 112, device='cuda', dtype=torch.float16)
        d_input = torch.cat([input_tensor.half().cuda(), pad], dim=0).contiguous()
    else:
        d_input = input_tensor[:BATCH_SIZE].half().cuda().contiguous()

    d_output = torch.empty(BATCH_SIZE, 512, dtype=torch.float16, device='cuda')

    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    return d_output[:actual_bs].float()


def verify_correctness(model, engine, num_samples=100):
    """对比 PyTorch 和 TRT refit 后的输出"""
    model_fp16 = model.half().cuda()
    dummy = torch.randn(num_samples, 3, 112, 112, device='cuda')

    with torch.no_grad():
        feat_pytorch = model_fp16(dummy.half()).float()

    feat_trt = run_trt_inference(engine, dummy)

    # cosine similarity
    feat_pytorch_norm = F.normalize(feat_pytorch, dim=1)
    feat_trt_norm = F.normalize(feat_trt, dim=1)
    cos_sim = (feat_pytorch_norm * feat_trt_norm).sum(dim=1)

    print(f"  PyTorch vs TRT-Refit cosine similarity:")
    print(f"    mean: {cos_sim.mean():.8f}")
    print(f"    min:  {cos_sim.min():.8f}")
    return cos_sim.mean().item()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ckpt_path = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'

    print("=" * 60)
    print("TensorRT Refit 验证 (BN 融合方案)")
    print("=" * 60)

    # Step 1: 加载模型 + 融合 BN + 导出 ONNX
    print("\n[Step 1] 加载模型 & 融合 BN & 导出 ONNX")
    model = load_model(ckpt_path)

    print("  融合 BN...")
    fused_model = _manual_fuse_bn(copy.deepcopy(model))
    fused_model.eval()

    # 验证融合正确性
    dummy_check = torch.randn(4, 3, 112, 112, device='cuda')
    with torch.no_grad():
        out_orig = model.cuda()(dummy_check)
        out_fused = fused_model.cuda()(dummy_check)
    fuse_cos = F.cosine_similarity(out_orig, out_fused, dim=1).mean()
    print(f"  融合正确性 (orig vs fused cosine): {fuse_cos:.8f}")

    # 导出融合模型
    export_fused_onnx_fp16(fused_model)

    # Step 2: 构建权重名映射 (融合模型 state_dict → ONNX names)
    print("\n[Step 2] 构建融合模型 → ONNX 权重映射")
    weight_mapping = get_onnx_weight_map(fused_model)

    # Step 3: 构建 REFIT engine
    print("\n[Step 3] 构建 REFIT-enabled engine (一次性)")
    build_time = build_refit_engine(None)
    if build_time is None:
        print("构建失败，退出")
        sys.exit(1)

    # Step 4: 用融合权重 refit (验证)
    print("\n[Step 4] Refit 验证 (同一 checkpoint, 融合权重)")
    fused_state = fused_model.half().state_dict()
    fused_state_clean = {}
    for k, v in fused_state.items():
        if v.is_floating_point():
            fused_state_clean[k] = v
        # 跳过非浮点 (num_batches_tracked)
    engine, refit_time = refit_engine_with_weights(fused_state_clean, weight_mapping)

    if engine is None:
        print("Refit 失败，退出")
        sys.exit(1)

    # Step 5: 正确性验证 (用融合模型对比)
    print("\n[Step 5] 正确性验证")
    cos_mean = verify_correctness(fused_model, engine, num_samples=100)

    if cos_mean < 0.99:
        print(f"  警告: cosine {cos_mean:.4f} 太低，映射可能有问题")
    else:
        print(f"  正确性验证通过!")

    # Step 6: 模拟多 checkpoint refit
    print("\n[Step 6] 模拟多 checkpoint refit")
    print("  流程: 加载新权重 → 融合 BN → 提取融合 state_dict → refit")

    state_dict_orig = model.state_dict()
    refit_times = []
    for i in range(3):
        # 模拟: 给原始模型加噪声 → 融合 → refit
        sim_model = copy.deepcopy(model)
        with torch.no_grad():
            for p in sim_model.parameters():
                if p.is_floating_point():
                    p.add_(torch.randn_like(p) * 0.001)
        sim_fused = _manual_fuse_bn(sim_model)
        sim_fused_state = sim_fused.half().state_dict()

        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(ENGINE_PATH, 'rb') as f:
            engine_fresh = runtime.deserialize_cuda_engine(memoryview(f.read()))
        refitter = trt.Refitter(engine_fresh, logger)

        refit_start = time.time()
        refittable = set(refitter.get_all_weights())
        count = 0
        for pt_key, onnx_name in weight_mapping.items():
            if onnx_name not in refittable:
                continue
            w = sim_fused_state.get(pt_key)
            if w is None or not w.is_floating_point():
                continue
            refitter.set_named_weights(onnx_name, trt.Weights(w.cpu().numpy()))
            count += 1
        success = refitter.refit_cuda_engine()
        t = time.time() - refit_start
        refit_times.append(t)
        print(f"  Checkpoint {i+1}: refit {count} weights in {t:.2f}s, success={success}")
        del engine_fresh, sim_model, sim_fused

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"  Engine 首次构建: {build_time:.1f}s")
    print(f"  单次 Refit 耗时: {np.mean(refit_times):.2f}s (avg of {len(refit_times)})")
    print(f"  BN 融合 + refit 总耗时: ~{np.mean(refit_times)+0.5:.1f}s/checkpoint")
    print(f"  正确性: cosine mean = {cos_mean:.8f}")
    print(f"\n  20 checkpoint 对比:")
    print(f"    每次重建: 20 × {build_time:.0f}s = {20*build_time/60:.1f} min")
    avg_refit = np.mean(refit_times) + 0.5  # 加上 BN 融合开销
    print(f"    Refit:   {build_time:.0f}s + 20 × {avg_refit:.1f}s = {(build_time + 20*avg_refit)/60:.1f} min")
    print(f"    节省: {(20*build_time - build_time - 20*avg_refit)/60:.1f} min")
    print("=" * 60)

    # 清理
    if os.path.exists(ONNX_PATH):
        os.remove(ONNX_PATH)

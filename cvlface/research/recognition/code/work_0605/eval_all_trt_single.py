"""
eval_all_3_single.py - TRT 多卡评估 (无 fabric/NCCL)

设计:
- 主进程构建 TRT engine (单卡, 一次性)
- 多进程并行: 每卡一个进程加载 engine 提取特征
- 数据分片: DataLoader + DistributedSampler (手动)
- 聚合: /dev/shm 写文件 → 主进程合并 → compute_metric
- 完全不依赖 fabric, 不触发 NCCL

由 eval_all_3_launcher.py 调用。
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os, sys, time, warnings
sys.path.append(os.path.join(root))
warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")

import numpy as np
np.bool = np.bool_

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
import sklearn.preprocessing
from tqdm import tqdm
from models import get_model
from general_utils.config_utils import load_config
from evaluations.custom_verification_evaluator import (
    compute_tpir_from_hist,
    find_tpir_at_far,
    generate_pairs_adaptive,
    IndexedDataset,
)
from evaluations.verifications.verification import calculate_roc2
from evaluations.cluster_utils import get_sim_matrix_large_scale_v4
from evaluations.ijbbc.evaluate import evaluate as ijbbc_evaluate
from evaluations.tinyface.evaluate import evaluate as tinyface_evaluate
from evaluations.custom_ijbbc_evaluator import get_pairs_data, compute_tpir_from_heap
from evaluations.cluster_utils import get_sim_matrix_batch_balanced_silent
from dataset.base_dataset import MXFaceDataset


BATCH_SIZE = 256
NUM_WORKERS = 5
SHM_DIR = '/dev/shm/eval3_trt'


def _collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    indexes = torch.tensor([e["index"] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels, "index": indexes}


class TRTInfer:
    """TRT FP16 推理器"""
    def __init__(self, engine_path, batch_size=256):
        import tensorrt as trt
        self.batch_size = batch_size
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
        self.context = self.engine.create_execution_context()
        # 按 tensor_mode 识别 input/output, 不依赖 index 顺序
        # (TRT 10/11 不保证 input 一定排在 index 0, 取反会导致输出全错)
        self.input_name, self.output_name = None, None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.input_name = n
            else:
                self.output_name = n
        assert self.input_name and self.output_name, 'IO tensor 识别失败'
        self.d_input = torch.zeros(batch_size, 3, 112, 112, dtype=torch.float16, device='cuda')
        self.d_output = torch.zeros(batch_size, 512, dtype=torch.float16, device='cuda')
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())
        # 用当前 stream, 保证 copy_ 与 execute 顺序一致 (避免跨 stream 竞态)
        self.stream = torch.cuda.current_stream()

    def __call__(self, x):
        total = x.shape[0]
        x_fp16 = x.half()
        if total <= self.batch_size:
            self.d_input[:total].copy_(x_fp16)
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            return self.d_output[:total].float()
        results = []
        for s in range(0, total, self.batch_size):
            e = min(s + self.batch_size, total)
            bs = e - s
            self.d_input[:bs].copy_(x_fp16[s:e])
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            results.append(self.d_output[:bs].float().clone())
        return torch.cat(results, dim=0)


def build_trt_engine(model, batch_size, cache_dir):
    """导出 ONNX → 构建 TRT FP16 engine"""
    import tensorrt as trt
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, 'model.onnx')
    engine_path = os.path.join(cache_dir, 'model.engine')

    model_fp16 = model.half().cuda()
    model_fp16.eval()
    dummy = torch.randn(batch_size, 3, 112, 112, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(model_fp16, dummy, onnx_path,
                          input_names=['input'], output_names=['output'],
                          opset_version=17, dynamo=False)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"TRT Error: {parser.get_error(i)}")
            return None
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    return engine_path


def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class HFIndexedDataset(torch.utils.data.Dataset):
    """HuggingFace Dataset 适配器 (用于 IJB-C 等 arrow 格式数据集)"""
    def __init__(self, data_path, transform, with_path=False):
        from datasets import Dataset as HFDataset
        self.dataset = HFDataset.load_from_disk(data_path)
        self.transform = transform
        self.with_path = with_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        pixel_values = self.transform(image)
        index = item['index']
        result = {"pixel_values": pixel_values, "index": index}
        if self.with_path:
            result["path"] = item.get('path', '')
        return result


def _collate_hf(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    indexes = torch.tensor([e["index"] for e in examples])
    return {"pixel_values": pixel_values, "index": indexes}


def _collate_tf(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    indexes = torch.tensor([e["index"] for e in examples])
    paths = [e["path"] for e in examples]
    return {"pixel_values": pixel_values, "index": indexes, "paths": paths}


def worker_extract_hf(rank, world_size, engine_path, dataset_path, shm_path):
    """HuggingFace Dataset 特征提取 worker (用于 ijbbc/ijbc_custom)"""
    torch.cuda.set_device(rank)

    transform = get_transform()
    dataset = HFIndexedDataset(dataset_path, transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, collate_fn=_collate_hf,
                            pin_memory=True, persistent_workers=True)

    infer = TRTInfer(engine_path, batch_size=BATCH_SIZE)

    all_feats_normal = []
    all_feats_flip = []
    all_index = []

    for batch in tqdm(dataloader, desc=f'[GPU {rank}]', disable=(rank != 0)):
        x = batch["pixel_values"].cuda(non_blocking=True)
        idx = batch["index"]

        x_flip = torch.flip(x, dims=[3])
        x_combined = torch.cat([x, x_flip], dim=0)

        with torch.no_grad():
            feats = infer(x_combined)

        B = x.shape[0]
        all_feats_normal.append(feats[:B].cpu())
        all_feats_flip.append(feats[B:].cpu())
        all_index.append(idx)

    result = {
        'features_normal': torch.cat(all_feats_normal, dim=0),
        'features_flip': torch.cat(all_feats_flip, dim=0),
        'index': torch.cat(all_index, dim=0),
    }
    save_path = os.path.join(shm_path, f'rank_{rank}.pt')
    torch.save(result, save_path)


def worker_extract_hf_tinyface(rank, world_size, engine_path, dataset_path, shm_path):
    """TinyFace 专用 worker: 需要保存 path 字段"""
    torch.cuda.set_device(rank)

    transform = get_transform()
    dataset = HFIndexedDataset(dataset_path, transform, with_path=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, collate_fn=_collate_tf,
                            pin_memory=True, persistent_workers=True)

    infer = TRTInfer(engine_path, batch_size=BATCH_SIZE)

    all_feats_normal = []
    all_feats_flip = []
    all_index = []
    all_paths = []

    for batch in tqdm(dataloader, desc=f'[GPU {rank}]', disable=(rank != 0)):
        x = batch["pixel_values"].cuda(non_blocking=True)
        idx = batch["index"]
        paths = batch["paths"]

        x_flip = torch.flip(x, dims=[3])
        x_combined = torch.cat([x, x_flip], dim=0)

        with torch.no_grad():
            feats = infer(x_combined)

        B = x.shape[0]
        all_feats_normal.append(feats[:B].cpu())
        all_feats_flip.append(feats[B:].cpu())
        all_index.append(idx)
        all_paths.extend(paths)

    result = {
        'features_normal': torch.cat(all_feats_normal, dim=0),
        'features_flip': torch.cat(all_feats_flip, dim=0),
        'index': torch.cat(all_index, dim=0),
        'paths': all_paths,
    }
    save_path = os.path.join(shm_path, f'rank_{rank}.pt')
    torch.save(result, save_path)


def gather_and_deduplicate_hf(shm_path, world_size):
    """合并 HF Dataset 提取的特征 (无 labels, 按 index 排序)"""
    all_data = []
    for rank in range(world_size):
        path = os.path.join(shm_path, f'rank_{rank}.pt')
        all_data.append(torch.load(path, map_location='cpu'))
        os.remove(path)

    features_normal = torch.cat([d['features_normal'] for d in all_data], dim=0)
    features_flip = torch.cat([d['features_flip'] for d in all_data], dim=0)
    index = torch.cat([d['index'] for d in all_data], dim=0)

    # 按 index 排序去重
    sorted_idx = torch.argsort(index)
    index = index[sorted_idx]
    features_normal = features_normal[sorted_idx]
    features_flip = features_flip[sorted_idx]

    # 去重 (DistributedSampler pad)
    unique_mask = torch.ones(len(index), dtype=torch.bool)
    unique_mask[1:] = index[1:] != index[:-1]
    features_normal = features_normal[unique_mask]
    features_flip = features_flip[unique_mask]
    index = index[unique_mask]

    return features_normal, features_flip, index


def gather_and_deduplicate_tinyface(shm_path, world_size):
    """合并 TinyFace 提取的特征 (含 paths, 按 index 排序)"""
    all_data = []
    for rank in range(world_size):
        path = os.path.join(shm_path, f'rank_{rank}.pt')
        all_data.append(torch.load(path, map_location='cpu'))
        os.remove(path)

    features_normal = torch.cat([d['features_normal'] for d in all_data], dim=0)
    features_flip = torch.cat([d['features_flip'] for d in all_data], dim=0)
    index = torch.cat([d['index'] for d in all_data], dim=0)
    # 收集 paths: 各 rank 的 paths 列表按顺序拼接
    all_paths = []
    for d in all_data:
        all_paths.extend(d['paths'])

    # 按 index 排序去重
    sorted_idx = torch.argsort(index)
    index = index[sorted_idx]
    features_normal = features_normal[sorted_idx]
    features_flip = features_flip[sorted_idx]
    all_paths = [all_paths[i] for i in sorted_idx.tolist()]

    # 去重
    unique_mask = torch.ones(len(index), dtype=torch.bool)
    unique_mask[1:] = index[1:] != index[:-1]
    features_normal = features_normal[unique_mask]
    features_flip = features_flip[unique_mask]
    index = index[unique_mask]
    mask_list = unique_mask.tolist()
    all_paths = [p for p, m in zip(all_paths, mask_list) if m]

    return features_normal, features_flip, index, all_paths


def compute_metric_tinyface(embeddings, image_paths, metadata_path):
    """TinyFace 评估: probe vs gallery identification"""
    meta = torch.load(metadata_path, weights_only=False)
    result = tinyface_evaluate(
        all_features=embeddings,
        image_paths=image_paths,
        meta=meta,
    )
    return result


def compute_metric_ijbc_custom(embeddings, real_indices, metadata_path, num_gpus):
    """IJB-C 自定义协议: template分组 + 多GPU相似度矩阵 + TPIR"""
    meta = torch.load(metadata_path, weights_only=False)

    # 将 tensor 转 numpy
    for k in meta:
        if torch.is_tensor(meta[k]):
            meta[k] = meta[k].numpy()

    # 计算 template 级别分组
    print("  计算 template 分组...")
    group_map = get_pairs_data(meta)
    templates = meta['templates']
    index_docid_list = [group_map[templates[i]] for i in range(len(templates))]
    print(f"  共有 {len(set(index_docid_list))} 个唯一 identity")

    # 使用 real_indices 映射到 query_ids
    query_ids = np.array([index_docid_list[idx] for idx in real_indices])

    # 全量计算
    target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 1e-5]

    N = len(query_ids)
    total_pairs = N * (N - 1) // 2
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs

    max_far = max(target_fars)
    topk = max(int(total_neg_pairs * max_far), 1000)

    print(f"  总对数: {total_pairs}, 正样本: {total_pos_pairs}, 负样本: {total_neg_pairs}")
    print(f"  维护 top-{topk} 负样本分数")

    pos_scores, neg_scores, _ = get_sim_matrix_batch_balanced_silent(
        query_feats_list=embeddings,
        query_ids=query_ids,
        num_gpus=num_gpus,
        block_size=2048 * 5,
        topk=topk,
        threshold=None,
        show_progress=True,
        return_stats_only=False,
        return_pairs_only=False
    )

    print(f"  正样本对: {len(pos_scores)}, 负样本对(topk): {len(neg_scores)}")

    result_all, _ = compute_tpir_from_heap(neg_scores, pos_scores, total_neg_pairs, target_fars)
    print(f"  全量结果: {result_all}")

    # 001 子集 (如果文件存在)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    list_001_path = os.path.join(script_dir, '001_ijbc_image_list.txt')
    if os.path.exists(list_001_path):
        with open(list_001_path, 'r') as f:
            image_list_001_names = [line.strip() for line in f.readlines()]

        # 建立映射
        real_idx_to_row = {}
        for row, idx in enumerate(real_indices):
            if idx not in real_idx_to_row:
                real_idx_to_row[idx] = row

        image_list_001_indices = []
        for img_name in image_list_001_names:
            try:
                file_idx = int(os.path.splitext(img_name)[0]) - 1
                if file_idx in real_idx_to_row:
                    image_list_001_indices.append(real_idx_to_row[file_idx])
            except (ValueError, KeyError):
                pass

        image_list_001_indices = np.array(image_list_001_indices)
        print(f"  001子集: {len(image_list_001_indices)} 张图片")

        if len(image_list_001_indices) > 0:
            image_feat_001 = embeddings[image_list_001_indices]
            query_ids_001 = query_ids[image_list_001_indices]

            N2 = len(query_ids_001)
            total_pairs_001 = N2 * (N2 - 1) // 2
            unique_ids_001, counts_001 = np.unique(query_ids_001, return_counts=True)
            total_pos_001 = sum(c * (c - 1) // 2 for c in counts_001)
            total_neg_001 = total_pairs_001 - total_pos_001
            topk_001 = max(int(total_neg_001 * max_far), 1000)

            pos_scores_001, neg_scores_001, _ = get_sim_matrix_batch_balanced_silent(
                query_feats_list=image_feat_001,
                query_ids=query_ids_001,
                num_gpus=num_gpus,
                block_size=2048 * 5,
                topk=topk_001,
                threshold=None,
                show_progress=True,
                return_stats_only=False,
                return_pairs_only=False
            )
            result_001, _ = compute_tpir_from_heap(
                neg_scores_001, pos_scores_001, total_neg_001, target_fars)
            print(f"  001子集结果: {result_001}")
        else:
            result_001 = {}
    else:
        result_001 = {}

    # 合并结果
    result = {}
    for key, value in result_all.items():
        result['all_' + key] = value
    for key, value in result_001.items():
        result['001_' + key] = value

    return result


def compute_metric_ijbbc(embeddings, metadata_path):
    """IJB-C 官方协议评估: 模板聚合 + pair verification + ROC"""
    meta = torch.load(metadata_path, weights_only=False)
    faceness_scores = meta['faceness_scores'].numpy() if torch.is_tensor(meta['faceness_scores']) else meta['faceness_scores']
    templates = meta['templates'].numpy() if torch.is_tensor(meta['templates']) else meta['templates']
    medias = meta['medias'].numpy() if torch.is_tensor(meta['medias']) else meta['medias']
    label = meta['label'].numpy() if torch.is_tensor(meta['label']) else meta['label']
    p1 = meta['p1'].numpy() if torch.is_tensor(meta['p1']) else meta['p1']
    p2 = meta['p2'].numpy() if torch.is_tensor(meta['p2']) else meta['p2']

    result = ijbbc_evaluate(embeddings, faceness_scores, templates, medias, label, p1, p2, dummy=False)
    return result


def worker_extract(rank, world_size, engine_path, dataset_path, shm_path):
    """每个 GPU 进程: 加载 engine → 提取本分片特征 → 保存到 /dev/shm"""
    torch.cuda.set_device(rank)

    # Load dataset
    transform = get_transform()
    rec_path = os.path.join(dataset_path, 'train.rec')
    if os.path.exists(rec_path):
        mx_dataset = MXFaceDataset(root_dir=dataset_path, local_rank=0)
        mx_dataset.transform = transform
    else:
        from torchvision.datasets import ImageFolder as TVImageFolder
        mx_dataset = TVImageFolder(dataset_path, transform=transform)

    dataset = IndexedDataset(mx_dataset)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, collate_fn=_collate_fn,
                            pin_memory=True, persistent_workers=True)

    # Load TRT engine
    infer = TRTInfer(engine_path, batch_size=BATCH_SIZE)

    # Extract features (normal + flip)
    all_feats_normal = []
    all_feats_flip = []
    all_labels = []
    all_index = []

    for batch in tqdm(dataloader, desc=f'[GPU {rank}]', disable=(rank != 0)):
        x = batch["pixel_values"].cuda(non_blocking=True)
        label = batch["labels"]
        idx = batch["index"]

        x_flip = torch.flip(x, dims=[3])
        x_combined = torch.cat([x, x_flip], dim=0)

        with torch.no_grad():
            feats = infer(x_combined)

        B = x.shape[0]
        all_feats_normal.append(feats[:B].cpu())
        all_feats_flip.append(feats[B:].cpu())
        all_labels.append(label)
        all_index.append(idx)

    # Save to /dev/shm
    result = {
        'features_normal': torch.cat(all_feats_normal, dim=0),
        'features_flip': torch.cat(all_feats_flip, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'index': torch.cat(all_index, dim=0),
    }
    save_path = os.path.join(shm_path, f'rank_{rank}.pt')
    torch.save(result, save_path)


def gather_and_deduplicate(shm_path, world_size):
    """主进程: 合并各 rank 的特征, 去重"""
    all_data = []
    for rank in range(world_size):
        path = os.path.join(shm_path, f'rank_{rank}.pt')
        all_data.append(torch.load(path, map_location='cpu'))
        os.remove(path)

    # Interleave (DistributedSampler 的分配方式)
    features_normal = torch.cat([d['features_normal'] for d in all_data], dim=0)
    features_flip = torch.cat([d['features_flip'] for d in all_data], dim=0)
    labels = torch.cat([d['labels'] for d in all_data], dim=0)
    index = torch.cat([d['index'] for d in all_data], dim=0)

    # 按 index 排序去重
    sorted_idx = torch.argsort(index)
    index = index[sorted_idx]
    features_normal = features_normal[sorted_idx]
    features_flip = features_flip[sorted_idx]
    labels = labels[sorted_idx]

    # 去重 (DistributedSampler pad 会导致最后几个重复)
    unique_mask = torch.ones(len(index), dtype=torch.bool)
    unique_mask[1:] = index[1:] != index[:-1]
    features_normal = features_normal[unique_mask]
    features_flip = features_flip[unique_mask]
    labels = labels[unique_mask]
    index = index[unique_mask]

    return features_normal, features_flip, labels


def compute_metric_type4(embeddings, query_ids, num_gpus):
    """type=4: large scale matrix + TPIR"""
    target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    pos_hist, neg_hist = get_sim_matrix_large_scale_v4(
        query_feats_list=embeddings,
        query_ids=query_ids,
        num_gpus=num_gpus,
        block_size=2048 * 2,
        show_progress=True,
    )
    result, thresholds = compute_tpir_from_hist(pos_hist, neg_hist, target_fars=target_fars)
    print('result:', result)
    print('thresholds:', thresholds)
    return result


def compute_metric_verification(embeddings, eval_data_path):
    """Verification 协议: pair-based (如 LFW, AgeDB-30, CFP-FP 等)
    数据集结构: 12000张图 → 6000对, is_same 标记每对是否同一人
    图片按对排列: [pair0_img0, pair0_img1, pair1_img0, pair1_img1, ...]
    """
    from datasets import Dataset as HFDataset
    ds = HFDataset.load_from_disk(eval_data_path)
    # is_same 每对取一个 (偶数索引)
    issame_list = np.array([ds[i]['is_same'] for i in range(0, len(ds), 2)])

    # embeddings 已按 index 排序, 拆成 pairs
    emb1 = embeddings[0::2]  # 偶数索引
    emb2 = embeddings[1::2]  # 奇数索引

    # L2 normalize
    emb1 = sklearn.preprocessing.normalize(emb1)
    emb2 = sklearn.preprocessing.normalize(emb2)

    # 计算每对的 L2 距离
    diff = emb1 - emb2
    dist = np.sum(diff ** 2, axis=1)

    thresholds = np.arange(0, 4, 0.01)
    _tpr, _fpr, accuracy = calculate_roc2(thresholds, dist, issame_list, nrof_folds=10)
    acc, std = np.mean(accuracy) * 100, np.std(accuracy) * 100
    result = {'acc': acc, 'std': std}
    return result


def compute_metric_default(embeddings, labels):
    """默认: generate_pairs_adaptive + calculate_roc2"""
    dist, issame_list = generate_pairs_adaptive(embeddings, labels)
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc2(thresholds, dist, issame_list, nrof_folds=1)
    accuracy = accuracy * 100
    acc, std = np.mean(accuracy), np.std(accuracy)
    x_labels = [1e-6, 1e-5, 1e-4, 1e-3]
    tpirs = find_tpir_at_far(tpr, fpr, target_fars=x_labels)
    result = {'acc': acc, 'std': std}
    for far, tpir in zip(x_labels, tpirs):
        result[f'tpir_at_far_{far}'] = tpir
    return result


def get_epoch_num(path):
    if 'epoch:' in path:
        filename = os.path.basename(path)
        try:
            return int(filename.split('_')[0].split(':')[1])
        except (IndexError, ValueError):
            return 0
    return 0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--eval_config_name', type=str, default='test_20260605')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--name', type=str, default='eval3')
    args = parser.parse_args()

    path = args.ckpt_path
    num_gpu = args.num_gpu
    epoch = get_epoch_num(path)
    print(f"评估 checkpoint: {os.path.basename(path)} (epoch={epoch})")

    # 加载模型并构建 TRT engine (主进程, GPU 0)
    torch.cuda.set_device(0)
    model_config = load_config(os.path.join(path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(path, 'model.pt'))
    model.eval()

    trt_cache = '/tmp/trt_eval3_cache'
    print("构建 TRT engine...")
    t0 = time.time()
    engine_path = build_trt_engine(model, batch_size=BATCH_SIZE, cache_dir=trt_cache)
    if engine_path is None:
        print("TRT 构建失败")
        sys.exit(1)
    print(f"TRT engine 构建: {time.time()-t0:.1f}s")
    del model
    torch.cuda.empty_cache()

    # 加载评估配置
    eval_config = load_config(f'evaluations/configs/{args.eval_config_name}.yaml')

    # 输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'eval3_results', args.name)
    os.makedirs(output_dir, exist_ok=True)

    all_result = {}

    for eval_name, info in eval_config.per_epoch_evaluations.items():
        eval_data_path = os.path.join(eval_config.data_root, info.path)
        eval_type = info.evaluation_type
        print(f"\n{'='*50}")
        print(f"评估: {eval_name} (type={eval_type})")
        print(f"{'='*50}")

        # /dev/shm 路径
        shm_path = os.path.join(SHM_DIR, eval_name)
        os.makedirs(shm_path, exist_ok=True)

        # 根据 eval_type 选择 worker
        if eval_type == 'tinyface':
            worker_fn = worker_extract_hf_tinyface
        elif eval_type in ('ijbbc', 'ijbc_custom', 'verification'):
            worker_fn = worker_extract_hf
        else:
            worker_fn = worker_extract

        # 多进程提取特征
        t0 = time.time()
        processes = []
        for rank in range(num_gpu):
            p = mp.Process(target=worker_fn,
                           args=(rank, num_gpu, engine_path, eval_data_path, shm_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 检查子进程退出码
        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"  GPU {failed} 提取失败!")
            continue

        extract_time = time.time() - t0
        print(f"  特征提取: {extract_time:.1f}s ({num_gpu} GPU)")

        # 聚合 & 计算指标
        t0 = time.time()
        metadata_path = os.path.join(eval_data_path, 'metadata.pt')

        if eval_type == 'verification':
            features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, num_gpu)
            print(f"  样本数: {len(index)}")
            embeddings = (features_normal + features_flip).numpy()
            result = compute_metric_verification(embeddings, eval_data_path)

        elif eval_type == 'ijbbc':
            features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, num_gpu)
            print(f"  样本数: {len(index)}")
            embeddings = (features_normal + features_flip).numpy()
            result = compute_metric_ijbbc(embeddings, metadata_path)

        elif eval_type == 'ijbc_custom':
            features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, num_gpu)
            print(f"  样本数: {len(index)}")
            embeddings = (features_normal + features_flip).numpy()
            embeddings = sklearn.preprocessing.normalize(embeddings)
            real_indices = index.numpy()
            result = compute_metric_ijbc_custom(embeddings, real_indices, metadata_path, num_gpus=num_gpu)

        elif eval_type == 'tinyface':
            features_normal, features_flip, index, image_paths = gather_and_deduplicate_tinyface(shm_path, num_gpu)
            print(f"  样本数: {len(index)}")
            embeddings = (features_normal + features_flip).numpy()
            result = compute_metric_tinyface(embeddings, image_paths, metadata_path)

        else:
            # custom_verification4, custom_verification, etc.
            features_normal, features_flip, labels = gather_and_deduplicate(shm_path, num_gpu)
            print(f"  样本数: {len(labels)}")
            embeddings = (features_normal + features_flip).numpy()
            embeddings = sklearn.preprocessing.normalize(embeddings)
            query_ids = labels.numpy()
            if eval_type in ('custom_verification4',):
                result = compute_metric_type4(embeddings, query_ids, num_gpus=num_gpu)
            else:
                result = compute_metric_default(embeddings, query_ids)

        print(f"  指标计算: {time.time()-t0:.1f}s")
        print(f"  结果: {result}")

        all_result.update({eval_name + "/" + k: v for k, v in result.items()})
        del features_normal, features_flip, embeddings

    # 保存结果 (与 torch 版本一致，经 summary() 转换 key 格式)
    all_result['epoch'] = epoch
    save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
    save_result.to_csv(os.path.join(output_dir, f'epoch_{epoch}_raw.csv'))

    from evaluations import summary
    mean, summary_dict = summary(save_result, epoch=epoch, step=0, n_images_seen=0)
    summary_result = pd.DataFrame(pd.Series(summary_dict), columns=['val'])
    summary_csv = os.path.join(output_dir, f'epoch_{epoch}_summary.csv')
    summary_result.to_csv(summary_csv)
    print(f"\n结果已保存: {summary_csv}")

    # 清理
    if os.path.exists(trt_cache):
        import shutil
        shutil.rmtree(trt_cache)
    if os.path.exists(SHM_DIR):
        import shutil
        shutil.rmtree(SHM_DIR, ignore_errors=True)

    print("评估完成!")

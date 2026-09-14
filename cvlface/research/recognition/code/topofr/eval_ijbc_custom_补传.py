"""
eval_ijbc_custom_补传.py - 专门评估 ijbc_custom 并补传缺失的 001 子集数据

用法:
python eval_ijbc_custom_补传.py \
  --num_gpu 8 \
  --eval_config_name test_20260605 \
  --ckpt_dir /data1/dataset_0605/train_output/s4_topofr_full_0605_09-04_0/checkpoints_every_epoch \
  --project_name work_0605_test \
  --name s4_topofr_full \
  --run_id <wandb_run_id>  # 可选，如果要补传到已有 run
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import sklearn.preprocessing
from tqdm import tqdm
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

from models import get_model
from general_utils.config_utils import load_config
from evaluations.custom_ijbbc_evaluator import get_pairs_data, compute_tpir_from_heap
from evaluations.cluster_utils import get_sim_matrix_batch_balanced_silent


BATCH_SIZE = 256
NUM_WORKERS = 5
SHM_DIR = '/dev/shm/eval_ijbc_补传'


def get_epoch_num(path):
    if 'epoch:' in path:
        filename = os.path.basename(path)
        try:
            epoch_part = filename.split('_')[0]
            return int(epoch_part.split(':')[1])
        except (IndexError, ValueError):
            return float('inf')
    if 'adaface' in path:
        filename = os.path.basename(path)
        try:
            epoch_part = filename.split('_')[-1]
            epoch_part = epoch_part.replace('epoch', '')
            return int(epoch_part)
        except (IndexError, ValueError):
            return float(0)
    return float(0)


def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class HFIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform):
        from datasets import Dataset as HFDataset
        self.dataset = HFDataset.load_from_disk(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        pixel_values = self.transform(image)
        index = item['index']
        return {"pixel_values": pixel_values, "index": index}


def _collate_hf(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    indexes = torch.tensor([e["index"] for e in examples])
    return {"pixel_values": pixel_values, "index": indexes}


class TRTInfer:
    """TRT 推理器"""
    def __init__(self, engine_path, batch_size=256):
        import tensorrt as trt
        self.batch_size = batch_size
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
        self.context = self.engine.create_execution_context()

        self.input_name, self.output_name = None, None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.input_name = n
            else:
                self.output_name = n
        assert self.input_name and self.output_name, 'IO tensor 识别失败'

        _trt2torch = {trt.float16: torch.float16, trt.float32: torch.float32}
        self.input_dtype = _trt2torch[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = _trt2torch[self.engine.get_tensor_dtype(self.output_name)]
        self.d_input = torch.zeros(batch_size, 3, 112, 112, dtype=self.input_dtype, device='cuda')
        self.d_output = torch.zeros(batch_size, 512, dtype=self.output_dtype, device='cuda')
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())
        self.stream = torch.cuda.current_stream()

    def __call__(self, x):
        total = x.shape[0]
        x_in = x.to(self.input_dtype)
        if total <= self.batch_size:
            self.d_input[:total].copy_(x_in)
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            return self.d_output[:total].float()
        results = []
        for s in range(0, total, self.batch_size):
            e = min(s + self.batch_size, total)
            bs = e - s
            self.d_input[:bs].copy_(x_in[s:e])
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            results.append(self.d_output[:bs].float().clone())
        return torch.cat(results, dim=0)


def build_trt_engine(model, batch_size, cache_dir, precision='fp16'):
    """导出 ONNX → 构建 TRT engine"""
    import tensorrt as trt
    if precision not in ('fp16', 'fp32'):
        raise ValueError(f"precision 仅支持 'fp16'/'fp32', 收到: {precision}")
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, 'model.onnx')
    engine_path = os.path.join(cache_dir, 'model.engine')

    use_fp16 = (precision == 'fp16')
    onnx_dtype = torch.float16 if use_fp16 else torch.float32
    export_model = (model.half() if use_fp16 else model.float()).cuda()
    export_model.eval()
    dummy = torch.randn(batch_size, 3, 112, 112, device='cuda', dtype=onnx_dtype)
    with torch.no_grad():
        torch.onnx.export(export_model, dummy, onnx_path,
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


def worker_extract_hf(rank, world_size, engine_path, dataset_path, shm_path):
    """HuggingFace Dataset 特征提取 worker"""
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


def gather_and_deduplicate_hf(shm_path, world_size):
    """合并 HF Dataset 提取的特征"""
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

    # 去重
    unique_mask = torch.ones(len(index), dtype=torch.bool)
    unique_mask[1:] = index[1:] != index[:-1]
    features_normal = features_normal[unique_mask]
    features_flip = features_flip[unique_mask]
    index = index[unique_mask]

    return features_normal, features_flip, index


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
    target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 1e-5, 1e-4, 1e-3]

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

    # 001 子集
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
        print(f"  警告: 未找到 {list_001_path}, 跳过 001 子集")
        result_001 = {}

    # 合并结果
    result = {}
    for key, value in result_all.items():
        result['all_' + key] = value
    for key, value in result_001.items():
        result['001_' + key] = value

    return result


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--eval_config_name', type=str, default='test_20260605')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--project_name', type=str, default="work_0605_test")
    parser.add_argument('--run_id', type=str, default=None, help="wandb run id，留空则创建新 run")
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'])
    args = parser.parse_args()

    # 找到所有 checkpoint
    checkpoint_path = args.ckpt_dir
    path_list = os.listdir(checkpoint_path)
    full_paths = [os.path.join(checkpoint_path, name) for name in path_list]
    sorted_paths = sorted(full_paths, key=get_epoch_num)

    print(f"共找到 {len(sorted_paths)} 个 checkpoint:")
    for p in sorted_paths:
        print(f"  epoch:{get_epoch_num(p)} - {os.path.basename(p)}")

    # 初始化 wandb
    import wandb
    if args.run_id:
        wandb_run = wandb.init(
            project=args.project_name,
            name=args.name,
            id=args.run_id,
            resume="must",
            dir=os.path.join(root, 'research/recognition/experiments', 'eval_ijbc_补传', args.name),
        )
        print(f"补传到已有 run: {args.run_id}")
    else:
        wandb_run = wandb.init(
            project=args.project_name,
            name=args.name + '_ijbc_custom',
            dir=os.path.join(root, 'research/recognition/experiments', 'eval_ijbc_补传', args.name),
        )
        print(f"创建新 run: {wandb_run.id}")

    # 加载评估配置
    eval_config = load_config(f'evaluations/configs/{args.eval_config_name}.yaml')

    # 找到 ijbc_custom 评估项
    ijbc_custom_name = None
    for eval_name, info in eval_config.per_epoch_evaluations.items():
        if info.evaluation_type == 'ijbc_custom':
            ijbc_custom_name = eval_name
            ijbc_custom_info = info
            break

    if not ijbc_custom_name:
        print("错误: 评估配置中未找到 ijbc_custom 类型的评估项")
        sys.exit(1)

    print(f"\n找到评估项: {ijbc_custom_name}")
    eval_data_path = os.path.join(eval_config.data_root, ijbc_custom_info.path)
    metadata_path = os.path.join(eval_data_path, 'metadata.pt')

    # 遍历每个 checkpoint
    for i, ckpt_path in enumerate(sorted_paths):
        epoch = get_epoch_num(ckpt_path)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sorted_paths)}] 评估: {os.path.basename(ckpt_path)} (epoch={epoch})")
        print(f"{'='*60}")

        # 加载模型并构建 TRT engine
        torch.cuda.set_device(0)
        model_config = load_config(os.path.join(ckpt_path, 'model.yaml'))
        model_config.start_from = ''
        model_config.freeze = False
        model = get_model(model_config)
        model.load_state_dict_from_path(os.path.join(ckpt_path, 'model.pt'))
        model.eval()

        trt_cache = f'/tmp/trt_ijbc_补传_{epoch}'
        print(f"构建 TRT engine (precision={args.precision})...")
        import time
        t0 = time.time()
        engine_path = build_trt_engine(model, batch_size=BATCH_SIZE, cache_dir=trt_cache,
                                       precision=args.precision)
        if engine_path is None:
            print("TRT 构建失败，跳过")
            continue
        print(f"TRT engine 构建: {time.time()-t0:.1f}s")
        del model
        torch.cuda.empty_cache()

        # /dev/shm 路径
        shm_path = os.path.join(SHM_DIR, f'epoch_{epoch}')
        os.makedirs(shm_path, exist_ok=True)

        # 多进程提取特征
        print("提取特征...")
        t0 = time.time()
        processes = []
        for rank in range(args.num_gpu):
            p = mp.Process(target=worker_extract_hf,
                           args=(rank, args.num_gpu, engine_path, eval_data_path, shm_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 检查子进程退出码
        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"  GPU {failed} 提取失败，跳过")
            continue

        extract_time = time.time() - t0
        print(f"  特征提取: {extract_time:.1f}s ({args.num_gpu} GPU)")

        # 聚合 & 计算指标
        print("计算指标...")
        t0 = time.time()
        features_normal, features_flip, index = gather_and_deduplicate_hf(shm_path, args.num_gpu)
        print(f"  样本数: {len(index)}")
        embeddings = (features_normal + features_flip).numpy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        real_indices = index.numpy()
        result = compute_metric_ijbc_custom(embeddings, real_indices, metadata_path, num_gpus=args.num_gpu)

        print(f"  指标计算: {time.time()-t0:.1f}s")
        print(f"  结果: {result}")

        # 上传到 wandb
        upload_dict = {'epoch': epoch}
        for k, v in result.items():
            upload_dict[f'{ijbc_custom_name}/{k}'] = v

        wandb.log(upload_dict)
        print(f"  已上传到 wandb")

        # 清理
        del features_normal, features_flip, embeddings
        import shutil
        if os.path.exists(trt_cache):
            shutil.rmtree(trt_cache)
        if os.path.exists(shm_path):
            shutil.rmtree(shm_path, ignore_errors=True)

    # 清理总目录
    if os.path.exists(SHM_DIR):
        import shutil
        shutil.rmtree(SHM_DIR, ignore_errors=True)

    wandb.finish()
    print('\n所有评估完成!')

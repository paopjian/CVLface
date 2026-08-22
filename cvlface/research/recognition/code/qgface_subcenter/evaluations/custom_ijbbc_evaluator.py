from datasets import Dataset
import torch
from functools import partial
from .base_evaluator import BaseEvaluator
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict
import time
import sklearn.preprocessing


def preprocess_transform(examples, image_transforms):
    images = [image.convert("RGB") for image in examples['image']]
    images = [image_transforms(image) for image in images]
    examples["pixel_values"] = images
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    indexes = torch.tensor([example["index"] for example in examples], dtype=torch.int)
    return {
        "pixel_values": pixel_values,
        "index": indexes,
    }


def get_pairs_data(meta):
    """从 eval_ijbc.ipynb 移植的函数,计算template级别的分组"""
    def get_group_map(meta):
        # 并查集获取所有template的唯一id,包括孤岛
        all_nodes = set(meta['p1']) | set(meta['p2'])

        class UnionFind:
            def __init__(self, nodes):
                self.parent = {x: x for x in nodes}
                self.rank = {x: 0 for x in nodes}
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1

        # 初始化并查集
        uf = UnionFind(all_nodes)

        # 合并 label == 1 的边
        for label, p1, p2 in zip(meta['label'], meta['p1'], meta['p2']):
            if label == 1:
                uf.union(p1, p2)

        # 生成 group_map(包含孤岛)
        group_map = {}
        root_to_id = {}
        for node in all_nodes:
            root = uf.find(node)
            if root not in root_to_id:
                root_to_id[root] = len(root_to_id)  # 自动递增 group_id
            group_map[node] = root_to_id[root]
        
        return group_map
    
    group_map = get_group_map(meta)
    return group_map


def compute_tpir_from_heap(neg_heap, pos_scores, total_neg_pairs, target_fars):
    """从堆中计算 TPIR"""
    # 将堆转换为排序数组(从高到低)
    neg_scores_sorted = sorted(neg_heap, reverse=True)
    
    results = {}
    thresholds = {}
    
    for far in target_fars:
        # 计算对应的索引
        idx = int(far * total_neg_pairs)
        
        if idx < len(neg_scores_sorted):
            threshold = neg_scores_sorted[idx]
        else:
            # 如果 FAR 太小,使用最小的负样本分数
            threshold = neg_scores_sorted[-1] if neg_scores_sorted else 0.0
        
        # 计算 TPIR
        tpir = np.mean(pos_scores >= threshold) if len(pos_scores) > 0 else 0.0
        
        results[f'tpir_at_far_{far}'] = float(tpir) * 100 
        thresholds[far] = threshold
    
    return results, thresholds


class CustomIJBCEvaluator(BaseEvaluator):
    def __init__(self, name, data_path, transform, fabric, batch_size, num_workers):
        super().__init__(name, fabric, batch_size)
        self.name = name
        self.data_path = data_path
        
        # 加载原始数据集
        raw_dataset = Dataset.load_from_disk(data_path)
        
        # 🔥 关键: 使用 dataset['index'] 建立映射
        # dataset['index'] 就是图片的真实索引,与 meta['templates'] 一一对应
        self.dataset_indices = raw_dataset['index']  # [0, 1, 2, ..., 469374]
        
        # 建立映射: 从 dataset 行号 -> 真实索引
        self.row_to_real_idx = {row: idx for row, idx in enumerate(self.dataset_indices)}
        
        # 如果需要根据文件名查找(作为备用)
        # 假设 dataset 中的 index 就是对应的真实序号
        self.real_idx_to_row = {idx: row for row, idx in enumerate(self.dataset_indices)}
        
        # 加载预处理后的数据集
        preprocess = partial(preprocess_transform, image_transforms=transform)
        dataset = raw_dataset.with_transform(preprocess)
        self.dataloader = fabric.setup_dataloader_from_dataset(
            dataset,
            is_train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        # 加载元数据
        self.meta = torch.load(os.path.join(data_path, 'metadata.pt'), weights_only=False)


    def integrity_check(self, eval_color_space, pipeline_color_space):
        assert eval_color_space == pipeline_color_space


    @torch.no_grad()
    def evaluate(self, pipeline, epoch=0, step=0, n_images_seen=0):
        pipeline.eval()
        collection = self.extract(pipeline)
        collection_flip = self.extract(pipeline, flip_images=True)
        if self.fabric.local_rank == 0:
            result = self.compute_metric(collection, collection_flip)
            self.log(result, epoch, step, n_images_seen)
        else:
            result = {}
        self.fabric.barrier()
        return result


    def extract(self, pipeline, flip_images=False):
        all_features = []
        all_index = []
        for batch_idx, batch in tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc=f'Custom IJBC Feature Extraction',
            disable=self.fabric.local_rank != 0
        ):
            batch = self.complete_batch(batch)

            if self.is_debug_run():
                if batch_idx > 10:
                    break

            images = batch['pixel_values']
            index = batch['index']

            if flip_images:
                images = torch.flip(images, dims=[3])
            
            features = pipeline(images)
            all_features.append(features.cpu().detach())
            all_index.append(index.cpu().detach())

        # aggregate across all gpus
        per_gpu_collection = {
            "index": torch.cat(all_index, dim=0),
            'features': torch.cat(all_features, dim=0)
        }

        # cpu based gathering
        collection = self.gather_collection(method='cpu', per_gpu_collection=per_gpu_collection)
        return collection


    def compute_metric(self, collection, collection_flip):
        if self.is_debug_run():
            return dummy_result

        # 计算 group_map 和 index_docid_list
        print("正在计算template分组...")
        self.group_map = get_pairs_data(self.meta)
        self.index_docid_list = [
            self.group_map[self.meta['templates'][i]] 
            for i in range(len(self.meta['templates']))
        ]
        print(f"完成分组计算,共有 {len(set(self.index_docid_list))} 个唯一identity")
        
        print('提取特征结束,进入计算阶段')
        
        # 合并正反面特征并归一化
        embeddings = (collection['features'] + collection_flip['features']).numpy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        
        # 🔥 关键: collection['index'] 就是真实索引,直接对应 meta['templates']
        real_indices = collection['index'].numpy()
        
        # 使用 index_docid_list 作为query_ids(template级别的分组)
        query_ids = np.array([self.index_docid_list[idx] for idx in real_indices])
        
        from .cluster_utils import get_sim_matrix_batch_balanced_silent
        
        # 1. 全量计算
        start = time.time()
        target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 1e-5]
        
        N = len(query_ids)
        total_pairs = N * (N - 1) // 2
        unique_ids, counts = np.unique(query_ids, return_counts=True)
        total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
        total_neg_pairs = total_pairs - total_pos_pairs
        
        max_far = max(target_fars)
        topk = max(int(total_neg_pairs * max_far), 1000)
        
        print(f"总对数: {total_pairs}, 正样本对数: {total_pos_pairs}, 负样本对数: {total_neg_pairs}")
        print(f"维护 top-{topk} 负样本分数")
        
        pos_scores, neg_scores, _ = get_sim_matrix_batch_balanced_silent(
            query_feats_list=embeddings,
            query_ids=query_ids,
            num_gpus=8,
            block_size=2048*5,
            topk=topk,
            threshold=None,
            show_progress=True,
            return_stats_only=False,
            return_pairs_only=False
        )
        
        print(f"计算矩阵耗时: {time.time() - start:.2f} 秒")
        print(f"正样本对数量: {len(pos_scores)}, 维护的负样本对数量: {len(neg_scores)}")
        
        result_1, thresholds = compute_tpir_from_heap(neg_scores, pos_scores, total_neg_pairs, target_fars)
        print(f"全量结果: {result_1}")
        
        # 2. 001 检测图片对比
        with open('./001_ijbc_image_list.txt', 'r') as f:
            lines = f.readlines()
        image_list_001_names = [line.strip() for line in lines]
        
        # 🔥 安全的索引转换
        image_list_001_indices = []
        missing_images = []
        
        # 构建哈希表加速查找
        real_idx_to_row = {}
        for row, idx in enumerate(real_indices):
            if idx not in real_idx_to_row:
                real_idx_to_row[idx] = row

        for img_name in tqdm(image_list_001_names, desc="映射001图片索引"):
            try:
                # 从文件名提取索引 (例如 "1.jpg" -> 0)
                file_idx = int(os.path.splitext(img_name)[0]) - 1
                
                # 在哈希表中查找对应的行号
                if file_idx in real_idx_to_row:
                    image_list_001_indices.append(real_idx_to_row[file_idx])
                else:
                    missing_images.append(img_name)

            except:
                missing_images.append(img_name)
        
        if missing_images:
            print(f"⚠️ 警告: {len(missing_images)} 张图片未找到映射")
            print(f"示例: {missing_images[:5]}")
        
        image_list_001_indices = np.array(image_list_001_indices)
        print(f"001子集: 共 {len(image_list_001_indices)} 张图片")
        
        # 提取对应的特征和标签
        image_feat_001 = embeddings[image_list_001_indices]
        query_ids_001 = query_ids[image_list_001_indices]
        
        # 计算001子集
        N = len(query_ids_001)
        total_pairs = N * (N - 1) // 2
        unique_ids, counts = np.unique(query_ids_001, return_counts=True)
        total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
        total_neg_pairs = total_pairs - total_pos_pairs
        
        topk = max(int(total_neg_pairs * max_far), 1000)
        
        print(f"001子集统计 - 总对数: {total_pairs}, 正样本: {total_pos_pairs}, 负样本: {total_neg_pairs}")
        
        pos_scores, neg_scores, _ = get_sim_matrix_batch_balanced_silent(
            query_feats_list=image_feat_001,
            query_ids=query_ids_001,
            num_gpus=8,
            block_size=2048*5,
            topk=topk,
            threshold=None,
            show_progress=True,
            return_stats_only=False,
            return_pairs_only=False
        )
        
        result_2, thresholds = compute_tpir_from_heap(neg_scores, pos_scores, total_neg_pairs, target_fars)
        print(f"001子集结果: {result_2}")
        
        # 合并结果
        result = {}
        for key, value in result_1.items():
            result['all_' + key] = value
        for key, value in result_2.items():
            result['001_' + key] = value
        
        return result


dummy_result = {
    'all_tpir_at_far_1e-10': 0.0,
    'all_tpir_at_far_1e-09': 0.0,
    'all_tpir_at_far_1e-08': 0.0,
    'all_tpir_at_far_1e-07': 0.0,
    'all_tpir_at_far_5e-07': 0.0,
    'all_tpir_at_far_1e-06': 0.0,
    'all_tpir_at_far_1e-05': 0.0,
    '001_tpir_at_far_1e-10': 0.0,
    '001_tpir_at_far_1e-09': 0.0,
    '001_tpir_at_far_1e-08': 0.0,
    '001_tpir_at_far_1e-07': 0.0,
    '001_tpir_at_far_5e-07': 0.0,
    '001_tpir_at_far_1e-06': 0.0,
    '001_tpir_at_far_1e-05': 0.0,
}

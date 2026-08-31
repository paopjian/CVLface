import random
import numpy as np
from datasets import Dataset
import torch
from functools import partial
from .base_evaluator import BaseEvaluator
from .verifications.verification import evaluate, calculate_roc2
import sklearn
from tqdm import tqdm
import os


import pandas as pd
import numbers
import atexit

from itertools import product
from collections import defaultdict

from dataset.base_dataset import MXFaceDataset
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import pickle
import time
import json
from PIL import Image, ImageDraw, ImageFont

def compute_tpir_from_heap(neg_heap, pos_scores, total_neg_pairs, target_fars):
    """
    从堆中计算 TPIR
    """
    # 将堆转换为排序数组（从高到低）
    neg_scores_sorted = sorted(neg_heap, reverse=True)
    
    results = {}
    thresholds = {}
    
    for far in target_fars:
        # 计算对应的索引
        idx = int(far * total_neg_pairs)
        
        if idx < len(neg_scores_sorted):
            threshold = neg_scores_sorted[idx]
        else:
            # 如果 FAR 太小，使用最小的负样本分数
            threshold = neg_scores_sorted[-1] if neg_scores_sorted else 0.0
        
        # 计算 TPIR
        tpir = np.mean(pos_scores >= threshold)if len(pos_scores) > 0 else 0.0
        
        results[f'tpir_at_far_{far}'] = float(tpir) * 100 
        thresholds[far] = threshold
    
    return results, thresholds


def compute_tpir_from_hist(pos_hist, neg_hist, hist_bins=20_000_000, hist_range=(-1.0, 1.0),
                           target_fars=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6]):
    """
    从直方图计算 TPIR
    pos_hist/neg_hist: shape (hist_bins,), 每个bin的计数
    """
    t0 = time.time()
    total_neg = int(neg_hist.sum())
    total_pos = int(pos_hist.sum())
    t_sum = time.time() - t0

    # 从右到左累积（高相似度到低相似度）
    # 注意 [::-1] 是负步长视图, cumsum 在非连续内存上可能显著变慢
    t0 = time.time()
    neg_cumsum = np.cumsum(neg_hist[::-1])
    pos_cumsum = np.cumsum(pos_hist[::-1])
    t_cumsum = time.time() - t0

    t0 = time.time()
    bin_edges = np.linspace(hist_range[0], hist_range[1], hist_bins + 1)
    t_edges = time.time() - t0

    t0 = time.time()
    results = {}
    thresholds = {}

    for far in target_fars:
        target_count = far * total_neg
        # neg_cumsum 单调递增, 找第一个 >= target_count 的位置
        idx = np.searchsorted(neg_cumsum, target_count, side='left')

        if idx >= hist_bins:
            threshold = hist_range[0]
            tpir = 1.0
        else:
            original_bin_idx = hist_bins - 1 - idx
            threshold = float(bin_edges[original_bin_idx])
            tpir = float(pos_cumsum[idx]) / total_pos if total_pos > 0 else 0.0

        results[f'tpir_at_far_{far}'] = float(tpir) * 100
        thresholds[far] = threshold
    t_search = time.time() - t0

    print(f"[timing] compute_tpir_from_hist: sum={t_sum:.2f}s "
          f"cumsum={t_cumsum:.2f}s linspace={t_edges:.2f}s search={t_search:.2f}s "
          f"({hist_bins:,} bins)")

    return results, thresholds


def compute_tpir_optimized(query_feats_list, query_ids, target_fars=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6]):
    """
    优化版本：只维护 top 1e-6 的负样本分数
    """
    
    N = len(query_ids)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    feats_t = torch.tensor(query_feats_list, dtype=torch.float32).to(device)
    ids_t = torch.tensor(query_ids).to(device)

    # 预计算总的负样本数量（基于上三角矩阵）
    total_pairs = N * (N - 1) // 2  # 上三角对数
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)  # 正样本对数
    total_neg_pairs = total_pairs - total_pos_pairs  # 负样本对数
    
    print(f"总对数: {total_pairs}, 正样本对数: {total_pos_pairs}, 负样本对数: {total_neg_pairs}")
    
    # 计算需要维护的 top-k 大小
    max_far = max(target_fars)
    top_k = max(int(total_neg_pairs * max_far), 1000)  # 至少保留1000个
    # print(f"维护 top-{top_k} 负样本分数")
    
    # 使用最小堆维护 top-k 负样本分数
    neg_heap = []
    pos_scores = []
    
    block_size = 2048*5
    start = time.time()
    
    for i in tqdm(range(0, N, block_size), desc='Processing blocks'):
        block1 = feats_t[i:i+block_size]
        ids1 = ids_t[i:i+block_size]

        for j in range(i, N, block_size):
            block2 = feats_t[j:j+block_size]
            ids2 = ids_t[j:j+block_size]

            # 相似度计算
            sim_block = block1 @ block2.T

            # 标签匹配
            label_eq = (ids1[:, None] == ids2[None, :])

            # 上三角 mask
            if i == j:
                mask = torch.triu(torch.ones_like(sim_block, dtype=torch.bool), diagonal=1)
            else:
                mask = torch.ones_like(sim_block, dtype=torch.bool)

            # 提取有效的相似度和标签
            flat_sim = sim_block[mask]
            flat_labels = label_eq[mask]

            # 正样本分数直接收集
            pos_sim = flat_sim[flat_labels]
            if len(pos_sim) > 0:
                pos_scores.append(pos_sim.half().cpu())

            # 负样本分数只保留 top-k
            neg_sim = flat_sim[~flat_labels]
            k_local = min(top_k * 2, len(neg_sim))  # 取稍多一点，避免遗漏
            topk_neg = neg_sim.topk(k_local).values
            neg_candidates = topk_neg.half().cpu().numpy()  # 传到 CPU
            
            if len(neg_heap) == 0:
                neg_heap = neg_candidates
            else:
                neg_heap = np.concatenate([neg_heap, neg_candidates])
            
            if len(neg_heap) > top_k:
                # O(n) 分区操作
                neg_heap = np.partition(neg_heap, -top_k)[-top_k:]
                # 可选：排序便于后续判断最小值
                neg_heap = np.sort(neg_heap)  # 升序，最小值在 [0]
            else:
                neg_heap = np.sort(neg_heap)  # 维持有序

    print(f"计算矩阵耗时: {time.time() - start:.2f} 秒")
    
    # 转换正样本分数
    start = time.time()
    pos_scores = torch.cat(pos_scores).numpy() if pos_scores else np.array([])
    print(f"正样本处理耗时: {time.time() - start:.2f} 秒")
    print(f"正样本对数量: {len(pos_scores)}, 维护的负样本对数量: {len(neg_heap)}")

    # 计算 TPIR
    start = time.time()
    result, thresholds = compute_tpir_from_heap(neg_heap, pos_scores, total_neg_pairs, target_fars)
    print(f"计算 TPIR 耗时: {time.time() - start:.2f} 秒")
    
    return result, thresholds


def get_image_merge(valid_image_paths, output_path, sub_sim_list, images_per_row=2, target_size=(112, 112),
                    no_id=True):
    # 设置文本区域高度和字体
    text_height = 20
    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # 可选更漂亮的字体
    except:
        font = ImageFont.load_default(font_size)

    # 计算网格尺寸
    total_images = len(valid_image_paths)
    rows = (total_images + images_per_row - 1) // images_per_row

    # 创建大图画布
    canvas_width = target_size[0] * images_per_row
    canvas_height = (target_size[1] + text_height) * rows
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    # 放置图片
    for i, img_path in enumerate(valid_image_paths):
        try:
            img = Image.open(img_path)
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            row = i // images_per_row
            col = i % images_per_row
            x = col * target_size[0]
            y = y = row * (target_size[1] + text_height)

            canvas.paste(img, (x, y))

            # 如果是第一张图片（id图片），添加绿框
            if i == 0 and not no_id:
                # 绘制绿色边框，线宽为5
                for offset in range(5):
                    draw.rectangle([x + offset, y + offset,
                                    x + target_size[0] - 1 - offset,
                                    y + target_size[1] - 1 - offset],
                                   outline='green')
            if i > 0:
                text_x = x
                text_y = y + target_size[1] + 2  # 图片下方空2像素开始写文字
                sim_text = str(sub_sim_list[i - 1])
                draw.text((text_x, text_y), sim_text, fill='black', font=font)
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue
    canvas.save(output_path, 'JPEG', quality=90)

def get_all_files(root, extension_list=['.jpg']):
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files

def save_images(image_list, out_dir, image_dir):
    image_list_all = get_all_files(image_dir)
    image_dict = {os.path.basename(image_path):image_path for image_path in image_list_all}
    os.makedirs(out_dir, exist_ok=True)
    for idx, row in enumerate(image_list):
        i = row['i']
        j = row['j']
        path_i = row['path_i']
        path_j = row['path_j']
        similarity = row['similarity']
        similarity = f"{similarity:.4f}"
        
        path_i = path_i.split('/')[-1]
        # doc_id_i = path_i.split('+')[0]
        # path_i = os.path.join(image_dir, doc_id_i,path_i)
        path_i = image_dict.get(path_i, None)
        
        path_j = path_j.split('/')[-1]
        # doc_id_j = path_j.split('+')[0]
        # path_j = os.path.join(image_dir, doc_id_j,path_j)
        path_j = image_dict.get(path_j, None)
        if path_i is None or path_j is None:
            print(f"图片未找到: {row['path_i']} 或 {row['path_j']}")
            continue
        doc_id_i = path_i.split('/')[-2]
        doc_id_j = path_j.split('/')[-2]
        # 进行图片合并
        get_image_merge([path_i,path_j],f"{out_dir}/{idx}+{doc_id_i}+{doc_id_j}.jpg",[similarity,similarity])
    
def get_image_pair(threshold,sim_matrix,labels,path_list):
    results = []
    n = sim_matrix.shape[0]
    for i in tqdm(range(n), desc="Processing rows"):
        # 提取第 i 行的相似度
        row = sim_matrix[i]  # shape: (N,)
        # 条件1: 相似度大于阈值
        high_sim = row > threshold
        # 条件2: label 不同
        label_mismatch = labels[i] != labels
        # 条件3: j > i （只取上三角，避免重复）
        upper_triangle = np.arange(n) > i
        
        # 同时满足
        valid_mask = high_sim & label_mismatch & upper_triangle
        
        j_indices = np.where(valid_mask)[0]
        j_sorted = j_indices
        for j in j_sorted:
            results.append({
                'i': int(i),
                'j': int(j),
                'label_i': int(labels[i]),
                'label_j': int(labels[j]),
                'similarity': float(sim_matrix[i, j]),
                'path_i': path_list[i],
                'path_j': path_list[j]
            })
    return results

def compute_tpir_at_far(neg_scores, pos_scores, target_fars=[1e-8, 1e-7, 1e-6]):
    """
    从磁盘读取负样本和正样本得分，计算指定 FAR 下的 TPIR
    """
    # 加载负样本得分并排序
    # neg_scores = np.memmap(neg_score_file, dtype=np.float32, mode='r')
    import time
    start_time = time.time()
    top1e6 = np.partition(neg_scores, -int(len(neg_scores)*1e-6))[-int(len(neg_scores)*1e-6):]
    top1e6 = np.sort(top1e6)[::-1]  # 从高到低排序
    # neg_sorted = np.sort(neg_scores)[::-1]  # 从高到低排序
    print(f"排序用时: {time.time()-start_time}s")
    # 加载正样本得分
    # pos_scores = np.memmap(pos_score_file, dtype=np.float32, mode='r')

    results = {}
    thresholds = {}
    num_neg = len(neg_scores)
    for far in target_fars:
        idx = max(int(far * num_neg), 0)
        idx = min(idx, len(top1e6) - 1)  # 确保索引不越界
        threshold = top1e6[idx]

        tpir = np.mean(pos_scores >= threshold)
        results[f'tpir_at_far_{far}'] = float(tpir)
        thresholds[far] = threshold
    return results, thresholds


def find_tpir_at_far(tpr, fpr, target_fars=[1e-6]):
    # 确保 fpr 和 tpr 是单调的
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    # 去重并保持单调性
    unique_fpr, unique_indices = np.unique(fpr_sorted, return_index=True)
   
    fpr_sorted = fpr_sorted[unique_indices]
    tpr_sorted = tpr_sorted[unique_indices]


    # 使用线性插值
    interp_func = interp1d(fpr_sorted, tpr_sorted, bounds_error=False, fill_value=(tpr_sorted[0], tpr_sorted[-1]), kind='nearest')
    results = []
    for far in target_fars:
        if far < unique_fpr.min():
            print(f"Target FAR ({far}) is below minimum observed FPR ({unique_fpr.min()}). ")
        # 计算 target_far 对应的 TPIR
        tpir = float(interp_func(far))
        results.append(tpir)
    return results


def generate_pairs_adaptive(features, labels, method='log', fixed_value=200, scale=100, base=2, max_cap=10000):
    label_to_indices = defaultdict(list)

    # Step 1: 构建 label -> indices 映射
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    dist_list = []
    issame_list = []

    all_indices = np.arange(len(labels))

    pos_count = 0
    neg_count = 0
    np.random.seed(42)  # 设置随机种子以确保可重复性
    # Step 2 & 3: 生成正负样本对
    for indices in tqdm(label_to_indices.values(), desc="Processing classes"):
        n = len(indices)
        indices = np.array(indices)

        # === 正样本对 ===
        if n >= 2:
            i_idx, j_idx = np.triu_indices(n, k=1)
            total_pos = len(i_idx)

            # 动态或固定上限
            if method == 'fixed':
                max_pairs = fixed_value
            elif method == 'log':
                max_pairs = min(int(scale * np.log(n) / np.log(base)), max_cap)
            else:
                raise ValueError("method must be 'fixed' or 'log'")

            num_pos = min(total_pos, max_pairs)
            selected = np.random.choice(total_pos, size=num_pos, replace=False)

            pos_i = indices[i_idx[selected]]
            pos_j = indices[j_idx[selected]]
            diff = features[pos_i] - features[pos_j]
            dist = np.sum(diff ** 2, axis=1)

            dist_list.append(dist)
            issame_list.extend([True] * len(dist))
            pos_count+=len(dist)

        # === 负样本对 ===
        neg_candidates = np.setdiff1d(all_indices, indices)
        if len(neg_candidates) == 0:
            continue

        num_neg = len(dist_list[-1])*10 if n >= 2 else 100  # 匹配正样本数*10，或默认100
        anchor_idx = np.random.choice(indices, size=num_neg, replace=True)
        neg_idx = np.random.choice(neg_candidates, size=num_neg, replace=True)

        diff = features[anchor_idx] - features[neg_idx]
        dist = np.sum(diff ** 2, axis=1)

        dist_list.append(dist)
        issame_list.extend([False] * len(dist))
        neg_count+=len(dist)

    # 合并结果
    distances = np.concatenate(dist_list, axis=0) if dist_list else np.array([])
    issame = np.array(issame_list)
    print(f"Generated {pos_count} positive pairs and {neg_count} negative pairs.")
    return distances, issame

def get_image_pairs_multi_threshold_fast(thresholds,query_feats_list, query_ids, path_list, block_size=2048,device='cuda:0',use_cosine=True  # 是否使用 cosine 相似度
):
    # 自动选择设备
    if device == 'cuda:auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 归一化特征（用于 cosine）
    feats = torch.tensor(query_feats_list, dtype=torch.float32)
    if use_cosine:
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    feats = feats.to(device)

    ids_t = torch.tensor(query_ids).to(device)
    path_list = np.array(path_list)
    N = len(query_feats_list)

    # 预分配结果：每个阈值一个列表（FP: label 不同 + 高相似；FN: label 相同 + 低相似）
    fp_results_dict = {th: [] for th in sorted(thresholds, reverse=True)}
    fn_results_dict = {th: [] for th in sorted(thresholds, reverse=True)}  # FN: 低于阈值但 label 相同 
    
    min_th = min(thresholds)
    max_th = max(thresholds)

    start = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, N, block_size), desc="Row blocks"):
            block1 = feats[i:i+block_size]
            ids1 = ids_t[i:i+block_size]
            B1 = block1.shape[0]

            for j in range(i, N, block_size):
                block2 = feats[j:j+block_size]
                ids2 = ids_t[j:j+block_size]
                B2 = block2.shape[0]

                # === GPU: 计算相似度 ===
                sim_block = block1 @ block2.T  # [B1, B2]

                # === GPU: 构造 mask ===
                label_mismatch = ids1[:, None] != ids2[None, :]  # [B1, B2]
                label_match = ids1[:, None] == ids2[None, :]       # [B1, B2] True if same label
                if i == j:
                    # 同 block：只取上三角（j > i）
                    triu_mask = torch.triu(torch.ones(B1, B2, dtype=torch.bool, device=device), diagonal=1)
                else:
                    triu_mask = torch.ones(B1, B2, dtype=torch.bool, device=device)

                # === 找 FP: label 不同 + sim > min_th（可能超过多个阈值）===
                fp_mask = label_mismatch & triu_mask & (sim_block > min_th)
                
                # === 找 FN: label 相同 + sim < max_th（可能低于多个阈值）===
                fn_mask = label_match & triu_mask & (sim_block < max_th)
                
                # # 合并 mask
                # combined_mask = label_mismatch & triu_mask & (sim_block > min_th)
                
                if fp_mask.sum().item() > 0:
                    ii_fp, jj_fp = torch.nonzero(fp_mask, as_tuple=True)
                    sim_fp = sim_block[ii_fp, jj_fp]

                    global_i_fp = (i + ii_fp).cpu()
                    global_j_fp = (j + jj_fp).cpu()
                    sims_fp = sim_fp.cpu().numpy()

                    for gi, gj, sim in zip(global_i_fp, global_j_fp, sims_fp):
                        for th in thresholds:
                            if sim >= th:
                                fp_results_dict[th].append({
                                    'i': int(gi),
                                    'j': int(gj),
                                    'label_i': int(query_ids[gi]),
                                    'label_j': int(query_ids[gj]),
                                    'similarity': float(sim),
                                    'path_i': path_list[gi],
                                    'path_j': path_list[gj]
                                })
                # 处理 FN
                # if fn_mask.sum().item() > 0:
                #     ii_fn, jj_fn = torch.nonzero(fn_mask, as_tuple=True)
                #     sim_fn = sim_block[ii_fn, jj_fn]

                #     global_i_fn = (i + ii_fn).cpu()
                #     global_j_fn = (j + jj_fn).cpu()
                #     sims_fn = sim_fn.cpu().numpy()

                #     for gi, gj, sim in zip(global_i_fn, global_j_fn, sims_fn):
                #         for th in thresholds:
                #             if sim < th:  # 注意这里是 <，表示低于该阈值才算 FN
                #                 fn_results_dict[th].append({
                #                     'i': int(gi),
                #                     'j': int(gj),
                #                     'label_i': int(query_ids[gi]),
                #                     'label_j': int(query_ids[gj]),
                #                     'similarity': float(sim),
                #                     'path_i': path_list[gi],
                #                     'path_j': path_list[gj]
                #                 })
                        
    print(f"✅ Done in {time.time() - start:.2f}s, device={device}")
    return {
        'fp': fp_results_dict,   # False Positives: 不同类但相似度过高
        'fn': fn_results_dict    # False Negatives: 同类但相似度过低
    }

# MXFaceDataset 没有返回idx, 需要包装一下返回index
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        return {
            "pixel_values": sample,
            "label": label,
            "index": idx  # 真实全局 index
        }

class CustomVerificationEvaluator(BaseEvaluator):
    def __init__(self, name, data_path, transform, fabric, batch_size, num_workers, type=None):
        super().__init__(name, fabric, batch_size)
        self.name = name
        self.batch_size = batch_size
        self.data_path = data_path

        # Support both RecordIO and ImageFolder formats
        rec_path = os.path.join(data_path, 'train.rec')
        idx_path = os.path.join(data_path, 'train.idx')
        if os.path.exists(rec_path) and os.path.exists(idx_path):
            self.mx_dataset = MXFaceDataset(root_dir=data_path, local_rank=0)
            self.mx_dataset.transform = transform
        else:
            from torchvision.datasets import ImageFolder as TVImageFolder
            self.mx_dataset = TVImageFolder(data_path, transform=transform)

        self.dataset = IndexedDataset(self.mx_dataset)
        # 需要使用collate_fn配合dataset组装batch
        def collate_fn(examples):
            pixel_values = torch.stack([e["pixel_values"] for e in examples])
            labels = torch.tensor([e["label"] for e in examples])
            indexes = torch.tensor([e["index"] for e in examples])
            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "index": indexes,
            }
        # self.dataloader = DataLoader(
        #     dataset=self.dataset,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     shuffle=False,
        #     drop_last=False,
        # )
        self.dataloader = fabric.setup_dataloader_from_dataset(self.dataset,
                                                               is_train=False,
                                                               batch_size=batch_size,
                                                               num_workers=num_workers,
                                                               collate_fn=collate_fn)
        self.type = type
        self.cached_embeddings = None
        self.cached_query_ids = None


    def integrity_check(self, eval_color_space, pipeline_color_space):
        assert eval_color_space == pipeline_color_space

    def _metric_sync_path(self, epoch, step, n_images_seen):
        """Return a per-job CPU control file for multi-GPU metric completion."""
        master_port = os.environ.get("MASTER_PORT", "default")
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "default")
        safe_name = self.name.replace(os.sep, "_").replace("/", "_")
        root = os.path.join(
            "/tmp",
            "qgface_metric_sync",
            f"{master_port}_{run_id}",
        )
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{safe_name}_{epoch}_{step}_{n_images_seen}.json")

    @staticmethod
    def _publish_metric_status(path, status, error=None):
        payload = {"status": status}
        if error is not None:
            payload["error"] = repr(error)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)

    @staticmethod
    def _wait_for_metric_status(path):
        timeout = float(os.environ.get("QGFACE_METRIC_SYNC_TIMEOUT_SEC", "7200"))
        deadline = time.monotonic() + timeout
        while not os.path.exists(path):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for metric status: {path}")
            time.sleep(0.1)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"Rank 0 metric failed: {payload.get('error', 'unknown error')}"
            )

    def run_rank_zero_metric(self, compute_fn, epoch, name_suffix, n_images_seen):
        """
        Execute compute_fn on rank 0 with status-file sync for other ranks.

        This method prevents NCCL deadlock when compute_fn uses multi-GPU workers
        (e.g., ThreadPoolExecutor with num_gpus=8). Non-zero ranks poll a status
        file instead of entering an NCCL barrier while rank 0's worker threads
        occupy their GPUs.

        Args:
            compute_fn: Callable that returns result dict, executed only on rank 0
            epoch: Current epoch number
            name_suffix: Unique suffix for the sync file (e.g., "combined_123")
            n_images_seen: Total images seen (for uniqueness)

        Returns:
            dict: Result from compute_fn (rank 0) or empty dict (other ranks)
        """
        # Generate unique sync file path
        metric_sync_path = self._metric_sync_path(epoch, name_suffix, n_images_seen)

        # Clean up old sync file before starting
        if self.fabric.local_rank == 0:
            try:
                os.remove(metric_sync_path)
            except FileNotFoundError:
                pass

        # Barrier before metric starts (all ranks ready)
        self.fabric.barrier()

        # Rank 0 executes multi-GPU metric, other ranks wait via file polling
        if self.fabric.local_rank == 0:
            try:
                start_time = time.time()
                result = compute_fn()
                elapsed = time.time() - start_time
                print(f"Rank 0 metric '{name_suffix}' completed in {elapsed:.2f}s")

                # Publish success status
                self._publish_metric_status(metric_sync_path, "ok")
            except Exception as error:
                # Publish error status to notify other ranks
                self._publish_metric_status(metric_sync_path, "error", error)
                raise
        else:
            result = {}
            # Poll status file instead of NCCL barrier
            self._wait_for_metric_status(metric_sync_path)

        # After metric completes, all ranks synchronize via NCCL
        self.fabric.barrier()
        torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def evaluate(self, pipeline, epoch=0, step=0, n_images_seen=0, save_image_path=None, save_pkl=None,image_dir=None):
        pipeline.eval()
        self.save_image_path = save_image_path
        self.image_dir = image_dir

        # type=4 launches an 8-GPU metric worker pool from rank 0.  Do not let
        # the other ranks enter an NCCL barrier while those GPUs are in use.
        metric_sync_path = self._metric_sync_path(epoch, step, n_images_seen) if self.type == '4' else None
        if metric_sync_path is not None:
            if self.fabric.local_rank == 0:
                try:
                    os.remove(metric_sync_path)
                except FileNotFoundError:
                    pass
            self.fabric.barrier()

        # 检查是否有缓存的pkl结果（所有rank都检查，避免分布式不一致）
        use_cache = save_pkl and os.path.exists(f'{save_pkl}/collection1.pkl')

        if use_cache:
            # 从缓存加载，不需要多GPU extract
            if self.fabric.local_rank == 0:
                with open(f'{save_pkl}/collection1.pkl','rb') as f:
                    collection = pickle.load(f)
                with open(f'{save_pkl}/collection2.pkl','rb') as f:
                    collection_flip = pickle.load(f)
        else:
            # 所有rank都必须参与extract（内部有分布式barrier同步）
            collection = self.extract(pipeline)
            collection_flip = self.extract(pipeline, flip_images=True)
            # rank 0 保存缓存
            if self.fabric.local_rank == 0 and save_pkl:
                os.makedirs(save_pkl, exist_ok=True)
                with open(f'{save_pkl}/collection1.pkl','wb') as f:
                    pickle.dump(collection, f)
                with open(f'{save_pkl}/collection2.pkl','wb') as f:
                    pickle.dump(collection_flip, f)

        if self.fabric.local_rank == 0:
            try:
                result = self.compute_metric(collection, collection_flip)
                self.log(result, epoch, step, n_images_seen)
                if metric_sync_path is not None:
                    self._publish_metric_status(metric_sync_path, "ok")
            except Exception as error:
                if metric_sync_path is not None:
                    self._publish_metric_status(metric_sync_path, "error", error)
                raise
            finally:
                del collection, collection_flip
        else:
            result = {}
            if not use_cache:
                del collection, collection_flip
            if metric_sync_path is not None:
                torch.cuda.empty_cache()
                self._wait_for_metric_status(metric_sync_path)

        # 等待 rank 0 的 compute_metric 完成，避免其多 GPU 计算与其他 rank 的下一轮 extract 冲突
        self.fabric.barrier()
        torch.cuda.empty_cache()
        return result

    def extract(self, pipeline, flip_images=False):
        all_features = []
        all_labels = []
        all_index = []
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                     desc=f'Verification {self.name}',
                                     disable=self.fabric.local_rank != 0):
            # batch = self.complete_batch(batch)  # needed for last batch to be gather compatible
            # x, label = batch
            x = batch["pixel_values"].to(self.fabric.device,non_blocking=True)
            label = batch["labels"]
            idx = batch["index"]
            
            # x = x.to('cuda')

            if self.is_debug_run():
                if batch_idx > 10:
                    break
            # if batch_idx > 10:
            #     break

            if flip_images:
                x = torch.flip(x, dims=[3])

            features = pipeline(x)
            all_features.append(features.cpu().detach())
            all_labels.append(label.cpu().detach())
            # all_index.append(torch.asarray([i for i in range(batch_idx*self.batch_size, batch_idx*self.batch_size+len(label))]).detach())
            all_index.append(idx.cpu().detach())

        # aggregate across all gpus
        per_gpu_collection = {"labels": torch.cat(all_labels, dim=0),
                              'features': torch.cat(all_features, dim=0),
                              "index": torch.cat(all_index, dim=0),
                              }
        # print(self.fabric.local_rank, per_gpu_collection)
        
        # cpu based gathering just in case we have a lot of data
        collection = self.gather_collection(method='cpu', per_gpu_collection=per_gpu_collection)
        
        torch.cuda.empty_cache()
        return collection


    def compute_metric(self, collection, collection_flip):
        if self.is_debug_run():
            print('Debug run, skipping metric computation')
            return {'acc': 0, 'std': 0}
        print('提取特征结束,进入计算阶段')
        # exit()
        embeddings = (collection['features'] + collection_flip['features']).numpy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        
        # 默认状态, 取部分做测试      
        if self.type == None:
            dist, issame_list = generate_pairs_adaptive(embeddings, collection['labels'].numpy())
            thresholds = np.arange(0, 4, 0.01)
            tpr,fpr,accuracy = calculate_roc2(thresholds, dist,issame_list, nrof_folds=1)
            accuracy = accuracy * 100
            acc, std = np.mean(accuracy), np.std(accuracy)
            # x_labels = [10 ** -8, 5 * 10 ** -8, 10 ** -7, 10 ** -6,10 ** -5,10 ** -4,10 ** -3]
            x_labels = [1e-6, 1e-5,1e-4,1e-3]
            tpirs = find_tpir_at_far(tpr, fpr, target_fars=x_labels)
            result = {'acc': acc, 'std': std}
            for far,tpir in zip(x_labels,tpirs):
                result[f'tpir_at_far_{far}'] = tpir

        if self.type == '2':
            pass
            query_feats_list = embeddings
            query_ids = collection['labels'].numpy()
            start = time.time()
            sim_matrix = np.dot(query_feats_list, query_feats_list.T).astype(np.float32)
            print(f"矩阵计算耗时: {time.time() - start:.2f} 秒")
            start = time.time()
            query_ids_expanded = query_ids[:, np.newaxis]  # shape: (N_q, 1)
            gallery_ids_expanded = query_ids[np.newaxis, :]  # shape: (1, N_g)
            labels_matrix = (query_ids_expanded == gallery_ids_expanded)
            print(f"labels_matrix计算耗时: {time.time() - start:.2f} 秒")
            start = time.time()
            mask = np.triu(np.ones_like(labels_matrix, dtype=bool), k=1)
            # 所有上三角位置（非对角）
            upper_indices = mask  # shape: (N, N), bool
            # 在上三角区域内，找出同类（正样本对）和不同类（负样本对）
            positive_mask = upper_indices & labels_matrix      # 上三角 && 同类
            negative_mask = upper_indices & (~labels_matrix)  # 上三角 && 不同类
            print(f"上三角掩码耗时: {time.time() - start:.2f}s")
            start = time.time()
            pos_scores = sim_matrix[positive_mask]
            neg_scores = sim_matrix[negative_mask]
            print(f"获取score 耗时: {time.time() - start:.2f}s")
            print(f"正样本对数量: {len(pos_scores)}, 负样本对数量: {len(neg_scores)}")
            
            start = time.time()
            x_labels = [10 ** -10, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6]
            result, thresholds = compute_tpir_at_far(neg_scores, pos_scores, target_fars=x_labels)
            end = time.time()
            print(result)
            print(thresholds)
            print(f"计算tpir耗时 : {end - start:.2f} 秒")

            if self.save_image_path:
                for far, threshold in thresholds.items():
                    image_list = get_image_pair(threshold, sim_matrix, query_ids, self.mx_dataset.info['path'].tolist())
                    save_path = self.save_image_path if self.save_image_path else f'/root/zhaokj/work/fp_merge/{int(time.time())}'
                    image_list = sorted(image_list, key=lambda x: x['similarity'], reverse=True)
                    path = os.path.join(save_path, f"{far}")
                    os.makedirs(path,exist_ok=True)
                    with open(os.path.join(save_path, f'results_{far}.txt'), 'w') as f:
                        f.write(f"Threshold: {threshold}\n")
                        for res in image_list:
                            f.write(f"{res['i']}\t{res['j']}\t{res['path_i']}\t{res['path_j']}\t{res['similarity']}\n")
                    save_images(image_list, path, self.image_dir)
       
        if self.type == '3':
            query_ids = collection['labels'].numpy()
            start = time.time()
            target_fars = [10 ** -10, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6]
            result, thresholds = compute_tpir_optimized(embeddings, query_ids, target_fars=target_fars)
            end = time.time()
            print('result: ',result)
            print('thresholds: ', thresholds)
            # print(f"计算tpir耗时 : {end - start:.2f} 秒")

            if self.save_image_path:
                thresholds_list = list(thresholds.values())
                thread_far = {v: k for k, v in thresholds.items()}
                results = get_image_pairs_multi_threshold_fast(
                    thresholds_list, embeddings, query_ids, self.mx_dataset.info['path'].tolist(),
                    device='cuda:0')
                results_fp = results['fp']
                results_fn = results['fn']
                
                image_list_all = get_all_files(self.image_dir)
                image_dict = {os.path.basename(image_path):image_path for image_path in image_list_all}
            
                for th, sub_results in results_fp.items():
                    save_path = os.path.join(self.save_image_path, 'fp')
                    os.makedirs(save_path, exist_ok=True)
                    sub_results = sorted(sub_results, key=lambda x: x['similarity'], reverse=True)
                    with open(os.path.join(save_path, f'results_{thread_far[th]}.txt'), 'w') as f:
                        f.write(f"Threshold: {th}\n")
                        tpir = result[f'tpir_at_far_{thread_far[th]}']
                        f.write(f"TPIR: {tpir}\n")
                        for res in sub_results:
                            path_i = res['path_i'].split('/')[-1]
                            path_i = image_dict.get(path_i, None)
                            path_j = res['path_j'].split('/')[-1]
                            path_j = image_dict.get(path_j, None)
                            if path_i is None or path_j is None:
                                print(f"图片未找到: {res['path_i']} 或 {res['path_j']}")
                                return None
                            f.write(f"{res['i']}\t{res['j']}\t{path_i}\t{path_j}\t{res['similarity']}\n")
                    save_images(sub_results, os.path.join(save_path, f'{thread_far[th]}'), self.image_dir)
                
                # for th, sub_results in results_fn.items():
                #     save_path = os.path.join(self.save_image_path, 'fn')
                #     os.makedirs(save_path, exist_ok=True)
                #     with open(os.path.join(save_path, f'results_{thread_far[th]}.txt'), 'w') as f:
                #         f.write(f"Threshold: {th}\n")
                #         tpir = result[f'tpir_at_far_{thread_far[th]}']
                #         f.write(f"TPIR: {tpir}\n")
                #         for res in sub_results:
                #             f.write(f"{res['i']}\t{res['j']}\t{res['path_i']}\t{res['path_j']}\t{res['similarity']}\n")
                #     save_images(sub_results, os.path.join(save_path, f'{thread_far[th]}'))
        
        if self.type == '4':
            # 使用 v4 直方图方式计算
            from .cluster_utils import get_sim_matrix_large_scale_v5

            query_ids = collection['labels'].numpy()
            start = time.time()

            target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]

            if self.save_image_path:
                # 先只算直方图，获取阈值
                pos_hist, neg_hist = get_sim_matrix_large_scale_v5(
                    query_feats_list=embeddings,
                    query_ids=query_ids,
                    num_gpus=8,
                    block_size=2048*4,
                    show_progress=True,
                )

                result, thresholds = compute_tpir_from_hist(pos_hist, neg_hist, target_fars=target_fars)
                print(f"计算矩阵+TPIR耗时: {time.time() - start:.2f} 秒")
                print('result: ', result)
                print('thresholds: ', thresholds)

                # 用最小阈值收集高相似度负样本对
                min_threshold = min(thresholds.values())
                print(f"正在获取高相似度图片对 (threshold >= {min_threshold})...")

                _, _, all_high_sim_pairs = get_sim_matrix_large_scale_v5(
                    query_feats_list=embeddings,
                    query_ids=query_ids,
                    num_gpus=8,
                    block_size=2048*2,
                    show_progress=True,
                    collect_pairs_config={
                        'sample_type': 'neg',
                        'threshold_mode': 'above',
                        'threshold': min_threshold,
                        'max_pairs': -1,
                    }
                )

                # 按阈值分组
                thresholds_list = list(thresholds.values())
                thread_far = {v: k for k, v in thresholds.items()}
                path_list = self.mx_dataset.info['path'].tolist()

                results_fp = defaultdict(list)
                for i, j, score in all_high_sim_pairs:
                    for th in thresholds_list:
                        if score >= th:
                            results_fp[th].append({
                                'i': int(i),
                                'j': int(j),
                                'label_i': int(query_ids[i]),
                                'label_j': int(query_ids[j]),
                                'similarity': float(score),
                                'path_i': path_list[i],
                                'path_j': path_list[j]
                            })

                image_list_all = get_all_files(self.image_dir)
                image_dict = {os.path.basename(image_path): image_path for image_path in image_list_all}

                for th, sub_results in results_fp.items():
                    save_path = os.path.join(self.save_image_path, 'fp')
                    os.makedirs(save_path, exist_ok=True)
                    sub_results = sorted(sub_results, key=lambda x: x['similarity'], reverse=True)

                    with open(os.path.join(save_path, f'results_{thread_far[th]}.txt'), 'w') as f:
                        f.write(f"Threshold: {th}\n")
                        tpir = result[f'tpir_at_far_{thread_far[th]}']
                        f.write(f"TPIR: {tpir}\n")
                        for res in sub_results:
                            path_i = res['path_i'].split('/')[-1]
                            path_i = image_dict.get(path_i, None)
                            path_j = res['path_j'].split('/')[-1]
                            path_j = image_dict.get(path_j, None)
                            if path_i is None or path_j is None:
                                print(f"图片未找到: {res['path_i']} 或 {res['path_j']}")
                                continue
                            f.write(f"{res['i']}\t{res['j']}\t{path_i}\t{path_j}\t{res['similarity']}\n")

                    save_images(sub_results, os.path.join(save_path, f'{thread_far[th]}'), self.image_dir)
            else:
                # 不需要保存图片，只算直方图和TPIR
                pos_hist, neg_hist = get_sim_matrix_large_scale_v5(
                    query_feats_list=embeddings,
                    query_ids=query_ids,
                    num_gpus=8,
                    block_size=2048*4,
                    show_progress=True,
                )

                result, thresholds = compute_tpir_from_hist(pos_hist, neg_hist, target_fars=target_fars)
                print(f"计算矩阵+TPIR耗时: {time.time() - start:.2f} 秒")
                print('result: ', result)
                print('thresholds: ', thresholds)

            # 缓存特征用于合并评估
            self.cached_embeddings = embeddings
            self.cached_query_ids = query_ids

        # result = {'acc': acc, 'std': std}
        if embeddings is not None:
            del embeddings
        # 必须 gc.collect 释放 ThreadPoolExecutor worker 在 GPU 1-6 上残留的张量
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return result



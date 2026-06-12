import os
import torch

from .verification_evaluator import VerificationEvaluator
from .ijbbc_evaluator import IJBBCEvaluator
from .tinyface_evaluator import TinyFaceEvaluator
from .custom_verification_evaluator import CustomVerificationEvaluator
from .custom_ijbbc_evaluator import CustomIJBCEvaluator
def get_evaluator_by_name(eval_type, name, eval_data_path, transform, fabric, batch_size, num_workers):

    assert os.path.isdir(eval_data_path), ('Evaluation Dataset does not exist. Check that cvlface/.env file is set correctly '
                                           'and the dataset is downloaded.')

    if eval_type == 'verification':
        return VerificationEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'ijbbc':
        return IJBBCEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'tinyface':
        return TinyFaceEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'ijbc_custom':
        return CustomIJBCEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type.startswith('custom_verification'):
        # 提取 custom_verification 后面的数字作为 type 参数
        if eval_type == 'custom_verification':
            type_param = None
        else:
            type_param = eval_type.replace('custom_verification', '')
        return CustomVerificationEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers, type=type_param)

    else:
        raise ValueError('Unknown evaluation type: %s' % eval_type)


def run_combined_evaluations(evaluators_dict, combined_config):
    """
    合并多个评估器的缓存特征，计算联合 TPIR 指标。
    仅在 rank 0 调用。

    Args:
        evaluators_dict: {name: evaluator} 评估器字典
        combined_config: 合并评估配置，如 {"work_0320": {"sources": ["work_0320_3t", "work_0320_glint"]}}

    Returns:
        dict: 合并后的评估结果，key 格式为 "combined_name/metric_name"
    """
    import numpy as np
    import time
    from .cluster_utils import get_sim_matrix_large_scale_v4
    from .custom_verification_evaluator import compute_tpir_from_hist

    all_combined_results = {}

    for combined_name, cfg in combined_config.items():
        sources = list(cfg.sources)

        # 检查所有源评估器是否都有缓存数据
        missing = [s for s in sources if s not in evaluators_dict
                   or not hasattr(evaluators_dict[s], 'cached_embeddings')
                   or evaluators_dict[s].cached_embeddings is None]
        if missing:
            print(f"跳过合并评估 '{combined_name}': 缺少源数据 {missing}")
            continue

        print(f"\n{'='*60}")
        print(f"开始合并评估: {combined_name} (sources: {sources})")
        print(f"{'='*60}")

        # 合并 embeddings 和 query_ids，使用 offset 防止 ID 冲突
        all_embeddings = []
        all_query_ids = []
        offset = 0
        for source_name in sources:
            evaluator = evaluators_dict[source_name]
            emb = evaluator.cached_embeddings
            ids = evaluator.cached_query_ids.copy()

            print(f"  {source_name}: {len(emb)} samples, "
                  f"{len(np.unique(ids))} classes, "
                  f"id range [{ids.min()}, {ids.max()}]")

            ids = ids + offset
            all_embeddings.append(emb)
            all_query_ids.append(ids)
            offset = int(ids.max()) + 1

        combined_embeddings = np.concatenate(all_embeddings, axis=0)
        combined_query_ids = np.concatenate(all_query_ids, axis=0)

        print(f"  合并后: {len(combined_embeddings)} samples, "
              f"{len(np.unique(combined_query_ids))} classes")

        # 计算合并评估
        start = time.time()
        target_fars = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]

        # _available_gpus = torch.cuda.device_count()
        _available_gpus = 7
        pos_hist, neg_hist = get_sim_matrix_large_scale_v4(
            query_feats_list=combined_embeddings,
            query_ids=combined_query_ids,
            num_gpus=_available_gpus,
            block_size=2048 * 2,
            show_progress=True,
        )

        result, thresholds = compute_tpir_from_hist(
            pos_hist, neg_hist, target_fars=target_fars
        )
        print(f"合并评估 '{combined_name}' 耗时: {time.time() - start:.2f} 秒")
        print(f"result: {result}")
        print(f"thresholds: {thresholds}")

        all_combined_results.update(
            {combined_name + "/" + k: v for k, v in result.items()}
        )

        # 清理合并数据
        del combined_embeddings, combined_query_ids

    # 清理源评估器缓存
    for evaluator in evaluators_dict.values():
        if hasattr(evaluator, 'cached_embeddings'):
            evaluator.cached_embeddings = None
            evaluator.cached_query_ids = None

    return all_combined_results


def summary(save_result, epoch, step, n_images_seen):
    key_metrics = ['cfpfp/acc', 'agedb_30/acc', 'lfw/acc',
                    'cplfw/acc', 'calfw/acc',
                    'tinyface/rank-1', 'tinyface/rank-5',
                    'IJBB_gt_aligned/Norm:False_Det:True_tpr_at_fpr_0.0001',
                    'IJBC_gt_aligned/Norm:False_Det:True_tpr_at_fpr_0.0001',
                    'work/acc',
                    'work/tpir_at_far_0.001',
                    'work/tpir_at_far_0.0001',
                    'work/tpir_at_far_1e-05',
                    'work/tpir_at_far_1e-06',
                    'work/tpir_at_far_1e-07',
                    'work/tpir_at_far_1e-08',
                    'work/tpir_at_far_1e-09',
                    'work/tpir_at_far_1e-10',
                    'work_1201/acc',
                    'work_1201/tpir_at_far_0.001',
                    'work_1201/tpir_at_far_0.0001',
                    'work_1201/tpir_at_far_1e-05',
                    'work_1201/tpir_at_far_1e-06',
                    'work_1201/tpir_at_far_1e-07',
                    'work_1201/tpir_at_far_1e-08',
                    'work_1201/tpir_at_far_1e-09',
                    'work_1201/tpir_at_far_1e-10',
                    'work_0213/acc',
                    'work_0213/tpir_at_far_0.001',
                    'work_0213/tpir_at_far_0.0001',
                    'work_0213/tpir_at_far_1e-05',
                    'work_0213/tpir_at_far_1e-06',
                    'work_0213/tpir_at_far_1e-07',
                    'work_0213/tpir_at_far_1e-08',
                    'work_0213/tpir_at_far_1e-09',
                    'work_0213/tpir_at_far_1e-10',
                    'work_0320/acc',
                    'work_0320/tpir_at_far_0.001',
                    'work_0320/tpir_at_far_0.0001',
                    'work_0320/tpir_at_far_1e-05',
                    'work_0320/tpir_at_far_1e-06',
                    'work_0320/tpir_at_far_1e-07',
                    'work_0320/tpir_at_far_1e-08',
                    'work_0320/tpir_at_far_1e-09',
                    'work_0320/tpir_at_far_1e-10',
                    'work_0320_3t/acc',
                    'work_0320_3t/tpir_at_far_0.001',
                    'work_0320_3t/tpir_at_far_0.0001',
                    'work_0320_3t/tpir_at_far_1e-05',
                    'work_0320_3t/tpir_at_far_1e-06',
                    'work_0320_3t/tpir_at_far_1e-07',
                    'work_0320_3t/tpir_at_far_1e-08',
                    'work_0320_3t/tpir_at_far_1e-09',
                    'work_0320_3t/tpir_at_far_1e-10',
                    'work_0320_glint/acc',
                    'work_0320_glint/tpir_at_far_0.001',
                    'work_0320_glint/tpir_at_far_0.0001',
                    'work_0320_glint/tpir_at_far_1e-05',
                    'work_0320_glint/tpir_at_far_1e-06',
                    'work_0320_glint/tpir_at_far_1e-07',
                    'work_0320_glint/tpir_at_far_1e-08',
                    'work_0320_glint/tpir_at_far_1e-09',
                    'work_0320_glint/tpir_at_far_1e-10',
                    'ijbc/all_tpir_at_far_1e-10',
                    'ijbc/all_tpir_at_far_1e-09',
                    'ijbc/all_tpir_at_far_1e-08',
                    'ijbc/all_tpir_at_far_1e-07',
                    'ijbc/all_tpir_at_far_5e-07',
                    'ijbc/all_tpir_at_far_1e-06',
                    'ijbc/all_tpir_at_far_1e-05',
                    'ijbc/001_tpir_at_far_1e-10',
                    'ijbc/001_tpir_at_far_1e-09',
                    'ijbc/001_tpir_at_far_1e-08',
                    'ijbc/001_tpir_at_far_1e-07',
                    'ijbc/001_tpir_at_far_5e-07',
                    'ijbc/001_tpir_at_far_1e-06',
                    'ijbc/001_tpir_at_far_1e-05',
                   ]
    key_metrics_in_save_result = [k for k in key_metrics if k in save_result.index]
    if key_metrics_in_save_result:
        summary = save_result.loc[key_metrics_in_save_result]
        summary.index = ['summary/'+k.replace('/', '_') for k in summary.index]
        summary.index = [k.replace('Norm:False_Det:True_tpr_at_fpr_0.0001', 'TPR@FPR0.01') for k in summary.index]
        summary.index = [k.replace('_gt_aligned', '') for k in summary.index]
        mean = summary['val'].mean()

        summary_dict = summary['val'].to_dict()
        summary_dict['epoch'] = epoch
        summary_dict['step'] = step
        summary_dict['n_images_seen'] = n_images_seen
        summary_dict['trainer/global_step'] = step
        summary_dict['trainer/epoch'] = epoch

    else:
        mean = save_result['val'].mean()
        summary_dict = save_result['val'].to_dict()
        summary_dict['epoch'] = epoch
        summary_dict['step'] = step
        summary_dict['n_images_seen'] = n_images_seen
        summary_dict['trainer/global_step'] = step
        summary_dict['trainer/epoch'] = epoch
    return mean, summary_dict


class IsBestTracker():

    def __init__(self, fabric):
        self._is_best = True
        self.prev_best_metric = -1
        self.fabric = fabric


    def set_is_best(self, metric):
        # metric = self.fabric.broadcast(metric, 0)
        metric_tensor = torch.tensor(metric, device=self.fabric.device)
        self.fabric.barrier()
        self.fabric.broadcast(metric_tensor, 0)
        metric = metric_tensor.item()

        if metric > self.prev_best_metric:
            self.prev_best_metric = metric
            self._is_best = True
        else:
            self._is_best = False


    def is_best(self):
        return self._is_best
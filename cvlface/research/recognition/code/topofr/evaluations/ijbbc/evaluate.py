import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.metrics import roc_curve


def image2template_feature(img_feats=None, templates=None, medias=None, dummy=False):
    """向量化版: (template, media) 组内均值 → template 内求和 → L2 归一化。

    等价于原逐 template/media 的 Python 循环, 但全部用 numpy 段聚合 (np.add.reduceat)。
    """
    img_feats = sklearn.preprocessing.normalize(img_feats) if False else img_feats
    if dummy:
        template_feats = np.random.randn(len(np.unique(templates)), img_feats.shape[1])
        return sklearn.preprocessing.normalize(template_feats), np.unique(templates)

    order = np.lexsort((medias, templates))
    t_s, m_s = np.asarray(templates)[order], np.asarray(medias)[order]
    feats_s = img_feats[order]

    # (template, media) 组边界
    new_group = np.r_[True, (t_s[1:] != t_s[:-1]) | (m_s[1:] != m_s[:-1])]
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.r_[starts, len(feats_s)]).astype(np.float32)[:, None]

    # media 组内均值 (ct==1 时即原特征, 与原实现等价)
    media_feats = np.add.reduceat(feats_s, starts, axis=0) / counts
    group_templates = t_s[starts]

    # template 段聚合 (media 组已按 template 排序, template 连续)
    t_change = np.r_[True, group_templates[1:] != group_templates[:-1]]
    unique_templates = group_templates[np.flatnonzero(t_change)]
    template_feats = np.add.reduceat(media_feats, np.flatnonzero(t_change), axis=0)
    return sklearn.preprocessing.normalize(template_feats), unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 500000  # 大批减少 Python 循环次数 (50 万对 fp32 中间约 1GB, 可承受)
    sublists = [ total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize) ]
    for c, s in tqdm(enumerate(sublists), total=len(sublists), desc='verification'):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
    return score



def evaluate(embeddings, faceness_scores, templates, medias, label, p1, p2, dummy=False):

    infernece_configs = [{'use_norm_score': True, 'use_detector_score': True},
                         {'use_norm_score': True, 'use_detector_score': False},
                         {'use_norm_score': False, 'use_detector_score': True}, ]

    scores = {}
    # media 分组结构只与 (templates, medias) 有关, 3 种配置共享一次计算
    order = np.lexsort((np.asarray(medias), np.asarray(templates)))
    t_s, m_s = np.asarray(templates)[order], np.asarray(medias)[order]
    new_group = np.r_[True, (t_s[1:] != t_s[:-1]) | (m_s[1:] != m_s[:-1])]
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.r_[starts, len(order)]).astype(np.float32)[:, None]
    group_templates = t_s[starts]
    t_change = np.r_[True, group_templates[1:] != group_templates[:-1]]
    unique_templates = group_templates[np.flatnonzero(t_change)]
    t_starts = np.flatnonzero(t_change)

    for config in infernece_configs:
        use_norm_score = config['use_norm_score']
        use_detector_score = config['use_detector_score']

        img_input_feats = embeddings.copy()
        if not use_norm_score:
            # normalise features to remove norm information
            img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))
        if use_detector_score:
            img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]

        feats_s = img_input_feats[order]
        media_feats = np.add.reduceat(feats_s, starts, axis=0) / counts
        template_feats = np.add.reduceat(media_feats, t_starts, axis=0)
        template_norm_feats = sklearn.preprocessing.normalize(template_feats)
        score = verification(template_norm_feats, unique_templates, p1, p2)
        method = f"Norm:{use_norm_score}_Det:{use_detector_score}"
        scores[method] = score

    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    result = {}
    for method in scores.keys():
        fpr, tpr, thresholds = roc_curve(label, scores[method])
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        thresholds = np.flipud(thresholds)
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            best_thresh = thresholds[min_index]
            _fpr_val = x_labels[fpr_iter]
            _tpr_val = tpr[min_index] * 100
            result[f'{method}_tpr_at_fpr_{_fpr_val}'] = _tpr_val
            result[f'{method}_thresh_at_fpr_{_fpr_val}'] = best_thresh
    return result

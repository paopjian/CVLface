#!/root/miniconda3/envs/cvlface/bin/python
"""
sheets3_s4.py — 新001 sheets3 三组选对 (fa_top50 / thr1e10_pm20 / thr1e9_above20)
补 s4(s4_0922_sub) 打分 + 按本报告 val_34t 阈值映射 FPIR 档, 重生成图墙。

Phase A: 从 feats_val34t.npz 重算 1e-3~1e-12 十档阈值 (tpir_fpir_full.json)
Phase B: 三 CSV 打 s4 分 + 档位 → sheets3_s4/*.csv + *_p*.png (008 同款布局)
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import io
import os
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluations.cluster_utils import get_sim_matrix_large_scale_v7
from evaluations.custom_verification_evaluator import compute_tpir_from_hist

V = '/root/zhaokj/work_eval/260922_val_34t'
REFS = '/data2/zkj-data/sub_fa/001_refs/sheets3_csv'
OUT = os.path.join(V, 'sheets3_s4')
DS_PATH = '/data1/dataset_260918/val_34t'
FONT = '/root/.local/share/fonts/SourceHanSansSC-Normal.otf'
F_BIG = ImageFont.truetype(FONT, 30)
F_MID = ImageFont.truetype(FONT, 26)
F_SM = ImageFont.truetype(FONT, 20)
IMG, GAP, ROW_H = 224, 12, 300
TXT_X_OFF = 10 + 2 * IMG + GAP + 18
W = TXT_X_OFF + 460
ROWS_PER_PAGE = 12

LEVELS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

# ---- Phase A: 十档阈值 ----
d = np.load('/data2/zkj-data/sub_fa/val34t/feats_val34t.npz')
feats = d['feats']
labels = d['labels'].astype(np.int64)
N = len(labels)
emb = feats.astype(np.float32)
print(f'载入特征 N={N:,}', flush=True)

pos_hist, neg_hist = get_sim_matrix_large_scale_v7(
    query_feats_list=emb, query_ids=labels, num_gpus=7, block_size=16384,
    hist_bins=2000, hist_range=(-1.0, 1.0))
result, thresholds = compute_tpir_from_hist(
    pos_hist, neg_hist, hist_bins=2000, hist_range=(-1.0, 1.0),
    target_fars=LEVELS)
print('十档 TPIR:', {f'{lv:g}': round(result[f'tpir_at_far_{lv}'], 2)
                     for lv in LEVELS}, flush=True)
print('十档阈值:', {f'{lv:g}': round(thresholds[lv], 4) for lv in LEVELS},
      flush=True)
with open(os.path.join(V, 'tpir_fpir_full.json'), 'w') as f:
    json.dump({'tpir': {f'{lv:g}': result[f'tpir_at_far_{lv}'] for lv in LEVELS},
               'thresholds': {f'{lv:g}': thresholds[lv] for lv in LEVELS},
               'total_neg_pairs': int(neg_hist.sum()),
               'total_pos_pairs': int(pos_hist.sum()),
               'n_samples': int(N)}, f, ensure_ascii=False, indent=2)

THR = {lv: thresholds[lv] for lv in LEVELS}

# 新001(133) val_34t 校准分阈值 (tpir_pos_exact.md) — 映射目标刻度
THR_001 = {1e-3: 0.63, 1e-4: 0.76, 1e-5: 0.80, 1e-6: 0.8395, 1e-7: 0.8555,
           1e-8: 0.8650, 1e-9: 0.8831, 1e-10: 0.9614, 1e-11: 0.9844,
           1e-12: 0.9946}
# 分段线性映射: s4 余弦 → 新001 校准刻度等效分 (锚点=同档阈值对)
_MAP_X = [THR[lv] for lv in LEVELS]      # 0.207 ... 0.555 升序
_MAP_Y = [THR_001[lv] for lv in LEVELS]  # 0.63 ... 0.9946


def s4_mapped(s):
    if s < _MAP_X[0]:
        return None                      # 低于 1e-3 档锚点, 无法可信映射
    return float(np.interp(s, _MAP_X, _MAP_Y))


def s4_mapped_txt(s):
    m = s4_mapped(s)
    return f'映射<{_MAP_Y[0]:g}' if m is None else f'映射≈{m:.3f}'


def s4_band(s):
    """按 val_34t 十档阈值映射: 返回 (档位文字, 颜色) — 最深过线档"""
    for lv in LEVELS[::-1]:                      # 深→浅
        if s >= THR[lv]:
            return f'≥{lv:g}线', '#cc0000'
    return f'<{LEVELS[0]:g}线', '#3a9950'


# ---- Phase B: 打分 + 图墙 ----
off = np.load(os.path.join(DS_PATH, 'train.offsets.npy'))
recf = open(os.path.join(DS_PATH, 'train.rec'), 'rb')


def read_img(idx):
    start = int(off[idx])
    end = int(off[idx + 1]) if idx + 1 < len(off) else -1
    recf.seek(start)
    raw = recf.read(end - start if end != -1 else -1)
    return Image.open(io.BytesIO(raw[32:])).convert('RGB')


def id_mask(s):
    return s[:6] + '*' * max(len(s) - 10, 0) + s[-4:] if len(s) >= 14 else s


def id_prefix(path):
    return os.path.basename(path).split('+')[0]


def build(name, title):
    df = pd.read_csv(os.path.join(REFS, f'{name}.csv'))
    a = df['rec_idx_a'].values.astype(np.int64)
    b = df['rec_idx_b'].values.astype(np.int64)
    df['s4_score'] = np.einsum('ij,ij->i', emb[a], emb[b])
    df['s4_band'] = [s4_band(s)[0] for s in df['s4_score']]
    df['s4_mapped'] = [s4_mapped(s) for s in df['s4_score']]
    df.to_csv(os.path.join(OUT, f'{name}_s4.csv'), index=False)

    n_pages = (len(df) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    for p in range(n_pages):
        chunk = df.iloc[p * ROWS_PER_PAGE:(p + 1) * ROWS_PER_PAGE]
        canvas = Image.new('RGB', (W, 44 + len(chunk) * ROW_H), '#eeeeee')
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 6), f'{title}  第{p+1}/{n_pages}页  共{len(df)}对',
                fill='#222222', font=F_MID)
        y = 44
        for _, r in chunk.iterrows():
            ia, ib = read_img(int(r['rec_idx_a'])), read_img(int(r['rec_idx_b']))
            canvas.paste(ia.resize((IMG, IMG), Image.BILINEAR), (10, y))
            canvas.paste(ib.resize((IMG, IMG), Image.BILINEAR), (10 + IMG + GAP, y))
            dr.rectangle([10, y, 10 + 2 * IMG + GAP, y + IMG - 1],
                         outline='#999999', width=1)
            band, color = s4_band(float(r['s4_score']))
            dr.text((TXT_X_OFF, y + 4),
                    f"#{int(r['rank'])}  新001={float(r['score']):.4f}",
                    fill='#222222', font=F_BIG)
            dr.text((TXT_X_OFF, y + 46),
                    f"s4={float(r['s4_score']):.4f}  ({s4_mapped_txt(float(r['s4_score']))})",
                    fill=color, font=F_BIG)
            dr.text((TXT_X_OFF, y + 88), f"s4档位: [{band}]",
                    fill=color, font=F_MID)
            o1 = r.get('old001_score')
            o_txt = (f"旧001={float(o1):.4f}" if pd.notna(o1)
                     else f"旧001=fail({r.get('old001_fail', '?')})")
            dr.text((TXT_X_OFF, y + 122), o_txt, fill='#0055aa', font=F_MID)
            dr.text((TXT_X_OFF, y + 156),
                    f"档案 {int(r['class_a'])}  {id_mask(id_prefix(r['name_a']))}",
                    fill='#333333', font=F_SM)
            dr.text((TXT_X_OFF, y + 182),
                    f"档案 {int(r['class_b'])}  {id_mask(id_prefix(r['name_b']))}",
                    fill='#333333', font=F_SM)
            dr.text((TXT_X_OFF, y + 208),
                    os.path.basename(r['name_a'])[:52], fill='#777777', font=F_SM)
            dr.text((TXT_X_OFF, y + 234),
                    os.path.basename(r['name_b'])[:52], fill='#777777', font=F_SM)
            dr.text((TXT_X_OFF, y + 260),
                    f"s4档位线: " + ' '.join(
                        f'{lv:g}:{THR[lv]:.3f}' for lv in
                        [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]),
                    fill='#999999', font=F_SM)
            y += ROW_H
        canvas.save(os.path.join(OUT, f'{name}_p{p+1}.png'))
    print(f'{name}: {n_pages} 页, s4分范围 [{df["s4_score"].min():.4f}, '
          f'{df["s4_score"].max():.4f}]', flush=True)


os.makedirs(OUT, exist_ok=True)
build('thr1e9_above20', '新001 1e-9阈值上方贴线20对 (+s4/旧001打分)')
build('thr1e10_pm20', '新001 1e-10阈值上下各20对 (+s4/旧001打分)')
build('fa_top50', '新001 FA Top50 (+s4/旧001打分)')
recf.close()
print('sheets3_s4 完成', flush=True)

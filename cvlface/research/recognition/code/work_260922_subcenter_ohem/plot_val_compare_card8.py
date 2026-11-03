#!/root/miniconda3/envs/cvlface/bin/python
"""
val_34t 三模型同图片对对比图 (card8 版): 新001(133实例, 已存档分数) / 旧001(old001_check.json)
/ subcenter(s4_0922_sub 本地)
输入: /root/zhaokj/work_eval/260922_val_34t/, 输出: 同目录 figs/ + summary_val.md
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

_FONT = '/root/.local/share/fonts/SourceHanSansSC-Normal.otf'
if os.path.isfile(_FONT):
    _fm.fontManager.addfont(_FONT)
    _FNAME = _fm.FontProperties(fname=_FONT).get_name()
else:
    _FNAME = 'sans-serif'
plt.rcParams["font.family"] = _FNAME
plt.rcParams['axes.unicode_minus'] = False

D = '/root/zhaokj/work_eval/260922_val_34t'
FIG = os.path.join(D, 'figs')
os.makedirs(FIG, exist_ok=True)

# 新001(133实例) val_34t 阈值/TPIR — tpir_pos_exact.md (2026-10-27 已存档, 不再打服务)
THR_001 = {1e-3: 0.63, 1e-4: 0.76, 1e-5: 0.80, 1e-6: 0.8395, 1e-7: 0.8555,
           1e-8: 0.8650, 1e-9: 0.8831, 1e-10: 0.9614, 1e-11: 0.9844, 1e-12: 0.9946}
TPIR_001 = {1e-3: 98.25, 1e-4: 97.36, 1e-5: 96.14, 1e-6: 94.47, 1e-7: 91.83,
            1e-8: 86.76, 1e-9: 72.33, 1e-10: 36.15, 1e-11: 10.86, 1e-12: 1.38}
# 旧001 val_34t TPIR 参照 (2026-10-04, README 存档)
TPIR_OLD = {1e-3: 98.59, 1e-4: 97.52, 1e-5: 95.65, 1e-6: 92.56, 1e-7: 86.42,
            1e-8: 77.44, 1e-9: 65.66, 1e-10: 45.74, 1e-11: 10.42, 1e-12: 1.05}

with open(os.path.join(D, 'tpir_fpir.json')) as f:
    sub = json.load(f)
THR_SUB = {float(k): v for k, v in sub['thresholds'].items()}
TPIR_SUB = {float(k.split('_')[-1]): v for k, v in sub['tpir'].items()}


def band_color(s):
    if s >= THR_001[1e-12]:
        return '#cc0000'
    if s >= THR_001[1e-11]:
        return '#cc6600'
    if s >= THR_001[1e-10]:
        return '#e6a23c'
    return '#4a7ebb'


def thr_lines(ax, v_lines=(), h_lines=(), vmapper=None, hmapper=None):
    for lv in v_lines:
        x = vmapper(lv)
        ax.axvline(x, color='#888', ls='--', lw=0.9)
        ax.text(x, 0.01, f'{lv:g}', rotation=90, fontsize=7.5,
                color='#555', transform=ax.get_xaxis_transform(), va='bottom')
    for lv in h_lines:
        y = hmapper(lv)
        ax.axhline(y, color='#4a7ebb', ls=':', lw=0.9)
        ax.text(0.995, y, f'{lv:g}', fontsize=7.5, color='#4a7ebb',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom')


# ================= fig1: 新001 FA 对(≥1e-9) 三模型对比 =================
fa = pd.read_csv(os.path.join(D, 'fa_pairs_1e-09.csv'))
colors = fa['score'].map(band_color)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
ax = axes[0]
ax.scatter(fa['score'], fa['sub_score'], s=9, c=colors, alpha=0.55, lw=0)
thr_lines(ax, v_lines=[1e-9, 1e-10],
          h_lines=[lv for lv in [1e-6, 1e-9, 1e-12] if lv in THR_SUB],
          vmapper=lambda lv: THR_001[lv], hmapper=lambda lv: THR_SUB[lv])
r = np.corrcoef(fa['score'], fa['sub_score'])[0, 1]
ax.set_xlabel('新001 相似度 (服务校准分)')
ax.set_ylabel('subcenter (s4_0922_sub) 余弦相似度')
ax.set_title(f'新001 FA 对 (≥1e-9, n={len(fa)}) 上两模型打分  r={r:.3f}')
ax.grid(alpha=0.25)

ax = axes[1]
m = fa['old001_score'].notna()
ax.scatter(fa.loc[m, 'score'], fa.loc[m, 'old001_score'], s=22,
           c=fa.loc[m, 'score'].map(band_color), alpha=0.8, lw=0,
           edgecolors='none', label='FA 对 (旧001 有分)')
ax.plot([0.85, 1.0], [0.85, 1.0], color='#999', lw=0.8, ls='-', label='y=x 参考')
ro = np.corrcoef(fa.loc[m, 'score'], fa.loc[m, 'old001_score'])[0, 1] \
    if m.sum() > 2 else float('nan')
ax.set_xlabel('新001 相似度 (服务校准分)')
ax.set_ylabel('旧001 余弦相似度')
ax.set_title(f'同 FA 对 旧001 打分 (n={int(m.sum())} 对覆盖)  r={ro:.3f}')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
fig.suptitle('val_34t 相同图片对: 新001 vs subcenter vs 旧001 — FA 对(跨ID高分负对)',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig1_fa_pairs.png'), dpi=150)
plt.close(fig)

# ================= fig2: 四类极值 top100 =================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(axA, axB), (axC, axD) = axes

neg_hi = pd.read_csv(os.path.join(D, 'neg_top100_001_cmp.csv'))
sub_neg_hi = pd.read_csv(os.path.join(D, 'neg_top100_high.csv'))
axA.scatter(neg_hi['score'], neg_hi['sub_score'], s=20, color='#4a7ebb',
            alpha=0.7, lw=0, label='新001 的 top100 对')
axA.set_title('(a) 负对相似度最高 top100')
axA.set_xlabel('新001 相似度')
axA.set_ylabel('subcenter 余弦')
axA.annotate(f'subcenter 自己的 neg-top100 范围\n'
             f'[{sub_neg_hi["score"].min():.3f}, {sub_neg_hi["score"].max():.3f}]',
             xy=(0.03, 0.05), xycoords='axes fraction', fontsize=9,
             color='#cc3333')
axA.legend(fontsize=8, loc='upper left')

sub_neg_lo = pd.read_csv(os.path.join(D, 'neg_top100_low.csv'))
axB.hist(sub_neg_lo['score'], bins=30, color='#4a7ebb', alpha=0.8)
axB.set_title('(b) 负对相似度最低 top100 — subcenter (新001 未存档)')
axB.set_xlabel('subcenter 余弦相似度')
axB.set_ylabel('对数')

sub_pos_hi = pd.read_csv(os.path.join(D, 'pos_top100_high.csv'))
axC.hist(sub_pos_hi['score'], bins=30, color='#3a9950', alpha=0.8)
axC.set_title('(c) 正对相似度最高 top100 — subcenter (新001 未存档)')
axC.set_xlabel('subcenter 余弦相似度')
axC.set_ylabel('对数')

pos_lo = pd.read_csv(os.path.join(D, 'pos_lowest_top100_001_cmp.csv'))
axD.scatter(pos_lo['score'], pos_lo['sub_score'], s=20, color='#3a9950',
            alpha=0.7, lw=0)
axD.set_title('(d) 新001 的正对最低 top100 对')
axD.set_xlabel('新001 相似度')
axD.set_ylabel('subcenter 余弦')

for ax in (axA, axB, axC, axD):
    ax.grid(alpha=0.25)
fig.suptitle('val_34t 四类极值对 (正/负 × 最高/最低, 各 top100)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig2_extremes.png'), dpi=150)
plt.close(fig)

# ================= fig3: 正对 near-miss (贴 1e-9 阈值下方 1000 对) =================
nm = pd.read_csv(os.path.join(D, 'pos_near_miss1000_001_cmp.csv'))
fig, ax = plt.subplots(figsize=(8, 6.8))
ax.scatter(nm['score'], nm['sub_score'], s=8, color='#3a9950', alpha=0.5, lw=0)
thr_lines(ax, v_lines=[1e-9],
          h_lines=[lv for lv in [1e-6, 1e-9, 1e-12] if lv in THR_SUB],
          vmapper=lambda lv: THR_001[lv], hmapper=lambda lv: THR_SUB[lv])
rn = np.corrcoef(nm['score'], nm['sub_score'])[0, 1]
ax.set_xlabel('新001 相似度 (服务校准分)')
ax.set_ylabel('subcenter 余弦相似度')
ax.set_title(f'新001 正对 near-miss (贴 1e-9 阈值下方, n={len(nm)})  r={rn:.3f}')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig3_pos_nearmiss.png'), dpi=150)
plt.close(fig)

# ================= 汇总 =================
pos_low_sub = pd.read_csv(os.path.join(D, 'pos_top100_low.csv'))
lines = ['# val_34t 三模型对比汇总 (相同图片对, 全部基于已存档分数+本地 s4_0922_sub)', '',
         '## TPIR@FPIR (各自阈值口径)', '',
         '| FPIR | 新001 TPIR% | 旧001 TPIR% | subcenter TPIR% | '
         '新001 阈值 | subcenter 阈值 |', '|---|---|---|---|---|---|']
for lv in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
    lines.append(f'| {lv:g} | {TPIR_001.get(lv, "")} | {TPIR_OLD.get(lv, "")} | '
                 f'{TPIR_SUB.get(lv, float("nan")):.2f} | {THR_001.get(lv, "")} | '
                 f'{THR_SUB.get(lv, float("nan")):.3f} |')
lines += ['', '## FA 对 / 极值统计', '',
          f'- 新001 FA 对 (≥1e-9): {len(fa)} 对; '
          f'新001 vs subcenter r={r:.3f}; 旧001 覆盖 {int(m.sum())} 对, r={ro:.3f}',
          f'- sub 在这些 FA 对上: mean={fa["sub_score"].mean():.3f}, '
          f'median={fa["sub_score"].median():.3f}, max={fa["sub_score"].max():.3f}, '
          f'≥sub阈值(1e-6)={100 * (fa["sub_score"] >= THR_SUB.get(1e-6, 9)).mean():.1f}%',
          f'- sub neg-top100 范围: [{sub_neg_hi["score"].min():.3f}, '
          f'{sub_neg_hi["score"].max():.3f}]; 新001 neg-top100 范围: '
          f'[{neg_hi["score"].min():.3f}, {neg_hi["score"].max():.3f}]',
          f'- sub pos-top100 范围: [{sub_pos_hi["score"].min():.3f}, '
          f'{sub_pos_hi["score"].max():.3f}]',
          f'- sub pos-lowest100 范围: [{pos_low_sub["score"].min():.3f}, '
          f'{pos_low_sub["score"].max():.3f}]; 新001 pos-lowest100 范围: '
          f'[{pos_lo["score"].min():.3f}, {pos_lo["score"].max():.3f}]',
          f'- sub neg-lowest100 范围: [{sub_neg_lo["score"].min():.3f}, '
          f'{sub_neg_lo["score"].max():.3f}]']
with open(os.path.join(D, 'summary_val.md'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('\nfigs ->', FIG)

"""把 card8_2 (run m20yly1y / subcenter_ohem_s2_c82) 的 epoch 10-17 结果
合并续写进主 run zgputcgd (subcenter_ohem_s2)。
数据源: eval3_results/subcenter_ohem_s2_c82/epoch_N_summary.csv (与 launcher 上传路径完全同源同格式)。
跑法: cd <code_dir> && https_proxy=http://127.0.0.1:7890 HTTPS_PROXY=... python merge_c82_into_main.py
"""
import os

import pandas as pd
import wandb

RES_DIR = 'eval3_results/subcenter_ohem_s2_c82'
MAIN_RUN_ID = 'zgputcgd'

rows = []
for e in range(10, 18):
    csv = os.path.join(RES_DIR, f'epoch_{e}_summary.csv')
    df = pd.read_csv(csv, index_col=0)
    d = df['val'].to_dict()
    rows.append((e, d))
    print(f'epoch {e}: {len(d)} 个指标, epoch字段={d.get("epoch")}')

run = wandb.init(project='work_260922_test', id=MAIN_RUN_ID, resume='must')
print('resumed run:', run.id, '起始 step:', run.step)
for e, d in rows:
    wandb.log(d)
    print(f'  logged epoch {e}')
run.finish()

# 验证
api = wandb.Api()
r = api.runs('work_260922_test', filters={'display_name': 'subcenter_ohem_s2'})[0]
h = r.history(keys=['epoch', '_step'], pandas=True).dropna(subset=['epoch'])
print('合并后主 run rows:', len(h))
print('epoch 字段:', sorted(int(x) for x in h['epoch']))
print('_step 值:', sorted(int(x) for x in h['_step']))

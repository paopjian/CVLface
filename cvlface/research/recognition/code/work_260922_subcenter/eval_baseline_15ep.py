"""
eval_baseline_15ep.py - 5 个基线模型在 qgface_subcenter 工作区 test/val 全集上的评估

用途: 给 work_260922_qgface_subcenter 的训练曲线提供可叠加的水平基线 (15 epochs)。
- 模型: subcenter_0605_e14 / s4_0618 / s3_full_0323_8 / s3_1226_14 / adaface_ir101_webface12m
- 配置: qgface 工作区 test_20260922 + val_20260922 的全部评估集
- wandb: 测试集 → project work_260922_test, 验证集 → project work_260922_val,
  run 名 = 模型名; 同一结果复制 15 个 epoch (0-14) —— 训练图上叠加即水平线
- TRT engine 缓存复用 (trt_engine_cache_baseline/): 每模型只编译一次 (~70s),
  同模型的 test/val 两次评估及重跑/续跑直接命中
- 断点续跑: <run_dir>/<config>/<model>.uploaded 存在则跳过; 只有 .json 时只补上传
- 加载门禁: 权重 key 兼容率 <90% 直接跳过 (防 strict=False 静默随机初始化)
- path 安全网: 若某评估集在 data_root 下不存在 (dataset_260918 的 facerec_val
  曾被清空过后恢复), 自动改写为 fallback 根 (/data1/dataset_0605 等) 下的绝对
  路径并打日志; 当前数据齐全, 正常情况零改写

用法:
LD_LIBRARY_PATH=/root/miniconda3/envs/cvlface/lib \
/root/miniconda3/envs/cvlface/bin/python eval_baseline_15ep.py [--dry_run] [--only 模型名]
"""
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ROOT = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition'
QG_CONFIG_DIR = ('/root/zhaokj/CVLface_rec/cvlface/research/recognition/code/'
                 'work_260922_qgface_subcenter/evaluations/configs')
MODELS = ['subcenter_0605_e14', 's4_0618', 's3_full_0323_8',
          's3_1226_14', 'adaface_ir101_webface12m']
# (config 名, wandb project)
CONFIGS = [('test_20260922', 'work_260922_test'),
           ('val_20260922', 'work_260922_val')]
FALLBACK_ROOTS = ['/data1/dataset_0605', '/data2/dataset_0320_rec']
N_EPOCH_LINES = 15


def has_ckpt(d):
    return os.path.isfile(os.path.join(d, 'model.pt')) and \
        os.path.isfile(os.path.join(d, 'model.yaml'))


def resolve_ckpt(name):
    """模型目录 → ckpt 目录; 顶层无 model.pt 时下钻唯一含 model.pt 的子目录
    (如 subcenter_0605_e14/epoch:14)。"""
    d = os.path.join(MODEL_ROOT, name)
    if not os.path.isdir(d):
        return None
    if has_ckpt(d):
        return d
    subs = [os.path.join(d, s) for s in sorted(os.listdir(d)) if has_ckpt(os.path.join(d, s))]
    return subs[0] if len(subs) == 1 else None


def check_load_compatible(ckpt, threshold=0.9):
    """CPU 上验证权重与 model.yaml 路由的 key 兼容率 (strict=False 加载器不兼容时
    静默随机初始化, 必须前置拦截)。"""
    import torch
    from general_utils.config_utils import load_config
    from models import get_model

    cfg = load_config(os.path.join(ckpt, 'model.yaml'))
    cfg.start_from = ''
    cfg.freeze = False
    model = get_model(cfg, 'work_0605')
    try:
        sd = torch.load(os.path.join(ckpt, 'model.pt'), map_location='cpu', weights_only=False)
        if hasattr(sd, 'state_dict'):
            sd = sd.state_dict()
        inter = len(set(sd.keys()) & set(model.state_dict().keys()))
        return inter / max(len(sd), 1), f'{inter}/{len(sd)}'
    finally:
        del model


def patch_config(config_name, patched_dir):
    """复制 qgface 配置到 patched_dir, 把 data_root 下不存在的相对 path 改写为
    fallback 根下的绝对路径。"""
    src = os.path.join(QG_CONFIG_DIR, f'{config_name}.yaml')
    with open(src) as f:
        text = f.read()
    data_root = re.search(r"data_root:\s*'([^']+)'", text).group(1).rstrip('/')

    rewrites = []
    for rel in sorted(set(re.findall(r"'path':\s*'([^']+)'", text))):
        if os.path.isabs(rel) or os.path.isdir(os.path.join(data_root, rel)):
            continue
        for fb in FALLBACK_ROOTS:
            if os.path.isdir(os.path.join(fb, rel)):
                text = text.replace(f"'path': '{rel}'", f"'path': '{os.path.join(fb, rel)}'")
                rewrites.append((rel, os.path.join(fb, rel)))
                break
        else:
            print(f'  [警告] {rel} 在 data_root 与 fallback 下都不存在, 该评估集会被跳过')

    dst = os.path.join(patched_dir, 'evaluations', 'configs', f'{config_name}.yaml')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        f.write(text)
    return dst, rewrites


def run_eval(model, ckpt, config_name, patched_dir, json_path, log_path, num_gpu, engine_cache_dir):
    """cwd=patched_dir 让 eval_all_trt_single.py 读到改写后的配置副本
    (配置路径是其相对 cwd 的 evaluations/configs/<name>.yaml)。"""
    ratio, detail = check_load_compatible(ckpt)
    if ratio < 0.9:
        return False, f'权重 key 兼容率 {ratio:.0%} ({detail})'

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'eval_all_trt_single.py'),
        '--num_gpu', str(num_gpu),
        '--eval_config_name', config_name,
        '--ckpt_path', ckpt,
        '--name', f'_qg_baseline/{config_name}/{model}',
        '--precision', 'fp16',
        '--result_path', json_path,
        '--engine_cache_dir', engine_cache_dir,
    ]
    env = os.environ.copy()
    env_prefix = os.path.dirname(os.path.dirname(sys.executable))
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpu))
    env.setdefault('CUDA_HOME', env_prefix)
    env['LD_LIBRARY_PATH'] = (os.path.join(env_prefix, 'lib') + os.pathsep
                              + env.get('LD_LIBRARY_PATH', ''))
    env['WANDB_MODE'] = 'disabled'
    env['PYTHONUNBUFFERED'] = '1'
    env['MPLCONFIGDIR'] = '/tmp/cvlface_qg_baseline_matplotlib'

    t0 = time.time()
    with open(log_path, 'a') as log_fh:
        log_fh.write(f"\n{'='*60}\n[{time.strftime('%F %T')}] cmd: {' '.join(cmd)}\n")
        log_fh.flush()
        try:
            proc = subprocess.run(cmd, env=env, cwd=patched_dir, timeout=120 * 60,
                                  stdout=log_fh, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            return False, '超时 (120min)'
    if proc.returncode != 0:
        return False, f'退出码 {proc.returncode}'
    if not os.path.isfile(json_path):
        return False, '成功退出但无 result JSON'
    print(f'  评估耗时 {time.time()-t0:.0f}s')
    return True, ''


def load_summary_metrics(model, config_name):
    """读 eval_all_trt_single 落盘的 epoch_*_summary.csv —— key 是 summary/* 格式,
    与训练/launcher 管线一致 (项目面板按这套 key 配置; 原始 all_result 的
    '数据集/指标' 带斜杠 key 不会出现在已有面板里, 别用)。"""
    import glob
    import pandas as pd
    csvs = glob.glob(os.path.join(SCRIPT_DIR, 'eval3_results', '_qg_baseline',
                                  config_name, model, 'epoch_*_summary.csv'))
    if not csvs:
        raise FileNotFoundError(f'{model}/{config_name} 无 epoch_*_summary.csv')
    df = pd.read_csv(csvs[0], index_col=0)
    return df['val'].to_dict()


def upload_15ep(model, ckpt, config_name, project, json_path, marker_path):
    """一模型一 run, 同一结果复制 15 个 epoch —— 训练曲线上叠加为水平基线。"""
    import wandb
    metrics = load_summary_metrics(model, config_name)

    run = wandb.init(
        project=project, name=model,
        config={'ckpt_path': os.path.realpath(ckpt),
                'eval_config': f'qgface_subcenter/{config_name}',
                'baseline_epochs': N_EPOCH_LINES},
    )
    try:
        for e in range(N_EPOCH_LINES):
            wandb.log(dict(metrics, epoch=e), step=e)
    finally:
        wandb.finish()
    with open(marker_path, 'w') as f:
        f.write(f'wandb run: {run.id} ({run.url})\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--only', type=str, default='', help='逗号分隔, 只评这些模型')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    engine_cache_dir = os.path.join(SCRIPT_DIR, 'trt_engine_cache_baseline')
    run_root = os.path.join(SCRIPT_DIR, 'eval3_results', '_qg_baseline')

    selected = [s.strip() for s in args.only.split(',')] if args.only else MODELS
    ckpts = {}
    for m in selected:
        ckpt = resolve_ckpt(m)
        if ckpt is None:
            print(f'未找到模型: {m}')
            sys.exit(1)
        ckpts[m] = ckpt

    print(f'模型 ({len(selected)}):')
    for m in selected:
        print(f'  + {m}  ({ckpts[m]})')
    for config_name, project in CONFIGS:
        patched_dir = os.path.join(run_root, config_name)
        dst, rewrites = patch_config(config_name, patched_dir)
        print(f'\n配置 {config_name} -> wandb project {project}')
        print(f'  副本: {dst}')
        for old, new in rewrites:
            print(f'  [path 改写] {old} -> {new}')
    if args.dry_run:
        return

    summary = []
    for config_name, project in CONFIGS:
        work_dir = os.path.join(run_root, config_name)
        os.makedirs(os.path.join(work_dir, 'logs'), exist_ok=True)
        for i, model in enumerate(selected):
            print(f"\n{'='*60}\n[{config_name}] ({i+1}/{len(selected)}) {model}\n{'='*60}",
                  flush=True)
            json_path = os.path.join(work_dir, f'{model}.json')
            marker_path = os.path.join(work_dir, f'{model}.uploaded')
            log_path = os.path.join(work_dir, 'logs', f'{model}.log')

            if os.path.isfile(marker_path):
                print('  已完成 (marker), 跳过')
                summary.append((config_name, model, 'done-previous'))
                continue
            if not os.path.isfile(json_path):
                ok, reason = run_eval(model, ckpts[model], config_name, work_dir,
                                      json_path, log_path, args.num_gpu, engine_cache_dir)
                if not ok:
                    tail = ''
                    try:
                        with open(log_path) as f:
                            tail = ''.join(f.readlines()[-8:]).rstrip()
                    except OSError:
                        pass
                    print(f'  ✗ 失败: {reason}\n{tail}')
                    summary.append((config_name, model, f'FAILED({reason})'))
                    continue
            try:
                upload_15ep(model, ckpts[model], config_name, project,
                            json_path, marker_path)
                print(f'  ✓ 已上传 wandb project={project} ({N_EPOCH_LINES} epochs 水平线)')
                summary.append((config_name, model, 'ok'))
            except Exception as e:
                print(f'  ✗ wandb 上传失败: {e}')
                summary.append((config_name, model, 'UPLOAD-FAILED'))

    print(f"\n{'='*60}\n汇总:\n{'='*60}")
    for config_name, model, status in summary:
        print(f'  [{status:>14}] {config_name}/{model}')


if __name__ == '__main__':
    main()

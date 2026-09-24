"""evaluate() 内部分阶段计时: 拆解 fabric 路径 105s 残差的构成.

不改仓库代码, 运行时给 evaluator 的 extract / gather_collection /
compute_metric 包计时; MXFaceDataset 构造(计时外)也单独计时。
启动: fabric run --devices=8 profile_eval_stages.py
"""
import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=['__root__.txt'],
                       pythonpath=True, dotenv=True)
import os
import sys
import time

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)

import datetime
import torch
from models import get_model
from aligners import get_aligner
from evaluations import get_evaluator_by_name
from pipelines import pipeline_from_name
from general_utils.config_utils import load_config
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset

CKPT = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'
DATA = '/data1/dataset_260918/val_glint'

STAGES = {}


def timed(fn, name):
    def wrap(*a, **k):
        t0 = time.time()
        r = fn(*a, **k)
        STAGES.setdefault(name, []).append(round(time.time() - t0, 2))
        return r
    return wrap


def main():
    torch.set_float32_matmul_precision('high')
    fabric = Fabric(precision='bf16-mixed', accelerator='auto',
                    strategy=DDPStrategy(timeout=datetime.timedelta(minutes=90)),
                    devices=8)
    if int(os.environ.get('LOCAL_RANK', -1)) == -1:
        fabric.launch()  # 仅单进程直跑时需要; fabric run CLI 已建进程
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset,
                                                   fabric=fabric, seed=2048)

    t0 = time.time()
    model_config = load_config(os.path.join(CKPT, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_260922')
    model.load_state_dict_from_path(os.path.join(CKPT, 'model.pt'))
    t_model = time.time() - t0

    aligner_config = load_config(os.path.join(
        pyrootutils.find_root(search_from=__file__, indicator=['__root__.txt']),
        'research/recognition/code/', 'run_v1', 'aligners/configs/none.yaml'))
    aligner = get_aligner(aligner_config)
    pipeline_name = load_config(os.path.join(CKPT, 'config.yaml')).pipelines.eval_pipeline_name
    model = fabric.setup(model)
    eval_pipeline = pipeline_from_name(pipeline_name, model, aligner)

    t0 = time.time()
    evaluator = get_evaluator_by_name(
        eval_type='custom_verification4', name='prof_val_glint',
        eval_data_path=DATA, transform=eval_pipeline.make_test_transform(),
        fabric=fabric, batch_size=256, num_workers=5)
    t_evaluator_init = time.time() - t0  # MXFaceDataset 解析 + DataLoader 构建 (计时外成本)

    evaluator.gather_collection = timed(evaluator.gather_collection, 'gather_cpu')

    # 细分: DataLoader 首批延迟( worker 启动) vs 迭代总时长
    class TimedDL:
        def __init__(self, dl):
            self.dl = dl

        def __len__(self):
            return len(self.dl)

        def __iter__(self):
            t0 = time.time()
            first = None
            for b in self.dl:
                if first is None:
                    first = time.time() - t0
                    STAGES.setdefault('dl_first_batch_s', []).append(round(first, 2))
                yield b
            STAGES.setdefault('dl_iter_total_s', []).append(round(time.time() - t0, 2))

    evaluator.dataloader = TimedDL(evaluator.dataloader)

    orig_empty = torch.cuda.empty_cache

    def timed_empty():
        t0 = time.time()
        orig_empty()
        STAGES.setdefault('empty_cache_s', []).append(round(time.time() - t0, 2))

    torch.cuda.empty_cache = timed_empty

    # 拆分: torch.cat (extract 循环后拼接) 与 pipeline forward
    orig_cat = torch.cat

    def timed_cat(*a, **k):
        t0 = time.time()
        r = orig_cat(*a, **k)
        STAGES.setdefault('torch_cat_s', []).append(round(time.time() - t0, 2))
        return r

    torch.cat = timed_cat

    class TimedPipeline:
        def __init__(self, p):
            self.p = p

        def __getattr__(self, name):
            return getattr(self.p, name)

        def __call__(self, x):
            t0 = time.time()
            r = self.p(x)
            STAGES.setdefault('forward_s', []).append(time.time() - t0)
            return r


    orig_extract = evaluator.extract

    def extract_with_loop_timing(pipeline, flip_images=False):
        t0 = time.time()
        r = orig_extract(pipeline, flip_images=flip_images)
        STAGES.setdefault(f'extract_total_{"flip" if flip_images else "normal"}',
                          []).append(round(time.time() - t0, 2))
        return r

    evaluator.extract = extract_with_loop_timing
    evaluator.compute_metric = timed(evaluator.compute_metric, 'compute_metric_total')

    t0 = time.time()
    result = evaluator.evaluate(TimedPipeline(eval_pipeline), epoch=0, step=0,
                                n_images_seen=0)
    t_eval = time.time() - t0

    if fabric.local_rank == 0:
        print('\n===== evaluate() 分阶段计时 (rank0) =====')
        print(f'模型加载(计时外):        {t_model:.1f}s')
        print(f'evaluator 构造(计时外):  {t_evaluator_init:.1f}s  '
              f'(MXFaceDataset tsv 解析 + DataLoader)')
        for k, v in STAGES.items():
            print(f'{k:>24}: {sum(v):.1f}s  {v}')
        print(f'evaluate() 总计:         {t_eval:.1f}s')
        print(f'  残差(barrier/pkl/empty_cache 等): '
              f'{t_eval - sum(sum(v) for v in STAGES.values()):.1f}s')
        print('result:', result)


if __name__ == '__main__':
    main()

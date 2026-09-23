"""单卡纯推理对比 (合成输入, 无 DataLoader, 不含编译时间): TRT fp16 vs torch eager/compile.

同 batch=512, 稳态吞吐 (warmup 后计时)。compile 用已热的 max-autotune
(bs512 首次会现场 autotune, 计时只从 warmup 完成后开始)。
"""
import os
import sys
import time

import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=['__root__.txt'],
                       pythonpath=True, dotenv=True)
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORK, 'portable_eval'))
sys.path.insert(0, WORK)

import numpy as np
import torch

MODEL_DIR = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'
ENGINE = os.path.join(WORK, 'bench_260922/engines/adaface_ir101_bs512_fp16.engine')
BS = 512
ITERS = 60
WARMUP = 20


def bench(fn, tag):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'{tag:>28}: {BS * ITERS / dt:,.0f} img/s  ({dt / ITERS * 1000:.2f} ms/批)')
    return BS * ITERS / dt


def main():
    from models import get_model
    from general_utils.config_utils import load_config
    import eval_lib as el

    torch.backends.cudnn.benchmark = True
    x16 = torch.randn(BS, 3, 112, 112, device='cuda', dtype=torch.float16)

    # 1. TRT fp16
    eng = el.TrtEngine(ENGINE, BS, device=0)
    bench(lambda: eng.infer(x16), 'TRT fp16 (bs512)')

    # 2. torch eager
    cfg = load_config(os.path.join(MODEL_DIR, 'model.yaml'))
    cfg.start_from = ''
    cfg.freeze = False
    model = get_model(cfg, 'work_260922').cuda().eval()

    with torch.no_grad():
        xf = x16.float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            bench(lambda: model(xf), 'torch eager bf16 (bs512)')

        mh = model.half()
        bench(lambda: mh(x16), 'torch eager fp16 (bs512)')

        # 3. torch compile (max-autotune) fp16 — 计时不含编译
        cm = torch.compile(mh, mode='max-autotune')
        cm(x16)  # 触发编译 (此处耗时不算)
        torch.cuda.synchronize()
        bench(lambda: cm(x16), 'compile max-autotune fp16')

        with torch.autocast('cuda', dtype=torch.bfloat16):
            cm(xf)
            torch.cuda.synchronize()
            bench(lambda: cm(xf), 'compile max-autotune bf16')


if __name__ == '__main__':
    main()

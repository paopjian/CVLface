"""方案 3 PoC: 线程分离 — engine 执行独立线程, 主线程专注 DataLoader 消费

机制: engine execute+sync 期间释放 GIL, 主线程的 iter/反序列化 (需 GIL) 得以
与 GPU 计算重叠, 消除 1:1 停顿. 理论上限 = min(纯供给 7,343, engine 7,420).

对比: 单进程基线 3,644 / 双进程 2,818 / DALI 3,297 (均为负收益或持平)
"""
import os
import queue
import sys
import threading
import time

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import cv2
from datasets import load_from_disk
from datasets.features.image import Image as HFImage

ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
ENGINE_BATCH = 256
DATA = '/data1/dataset_0605/facerec_val/IJBC_gt_aligned'
N_LIMIT = int(os.environ.get('POC_N', '120000'))


class DecodeDataset(torch.utils.data.Dataset):
    def __init__(self):
        ds = load_from_disk(DATA)
        self.ds = ds.cast_column('image', HFImage(decode=False))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        raw = bytes(self.ds[idx]['image']['bytes'])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy((img - 0.5) / 0.5).permute(2, 0, 1)


def main():
    torch.cuda.set_device(0)
    from opt_eval.tensorrt.int8_ptq_test import EngineRunner
    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)

    subset = DecodeDataset()
    loader = torch.utils.data.DataLoader(
        subset, batch_size=ENGINE_BATCH // 2, num_workers=10,
        collate_fn=lambda b: torch.stack(b), pin_memory=True,
        persistent_workers=True, prefetch_factor=4)

    q = queue.Queue(maxsize=4)
    done = threading.Event()
    n_eng = 0

    def engine_loop():
        nonlocal n_eng
        while True:
            item = q.get()
            if item is None:
                done.set()
                return
            runner(item)  # execute+sync 释放 GIL, 主线程可继续收批
            n_eng += 1

    eng_th = threading.Thread(target=engine_loop, daemon=True)
    eng_th.start()

    t0 = time.time()
    cnt = 0
    for xb in loader:
        x = xb.cuda(non_blocking=True)  # pinned 源, 异步 H2D
        xc = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        q.put(xc)
        cnt += xb.shape[0]
        if cnt >= N_LIMIT:
            break
    q.put(None)
    done.wait()
    dt = time.time() - t0
    print(f"线程分离: {cnt:,} 张 / {dt:.1f}s = {cnt/dt:,.0f} img/s "
          f"(单进程基线 3,644; 纯供给 7,343; engine 上限 7,420)")


if __name__ == '__main__':
    main()

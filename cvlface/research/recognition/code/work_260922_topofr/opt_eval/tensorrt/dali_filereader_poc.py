"""方案 1 PoC: DALI fn.readers.file 直读 cv4 JPEG 目录 (全 C++ 数据链路)

绕开 python 逐样本喂入: DALI C++ 线程读文件+解码+归一化, 输出直接给 engine.
数据: /data1/dataset_0605/test_enhance (20 万张 JPEG, 根/身份目录/图.jpg)
验证: 吞吐 (vs cv2 链 ~3,646) + label 与目录排序一致性 + 数值一致性

用法: python dali_filereader_poc.py
"""
import os
import sys
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

DATA = '/data1/dataset_0605/test_enhance'
ENGINE = '/tmp/int8_ptq_work/engine_fp16.plan'
ENGINE_BATCH = 256


def build_pipeline(device='cpu', num_threads=24):
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types

    pipe = dali.pipeline.Pipeline(batch_size=ENGINE_BATCH, num_threads=num_threads,
                                  device_id=0, prefetch_queue_depth=4)
    with pipe:
        jpegs, labels = fn.readers.file(file_root=DATA, random_shuffle=False)
        img = fn.decoders.image(jpegs, device=device, output_type=types.RGB)
        img = fn.cast(img, dtype=types.FLOAT) * (1 / 255.0)
        img = (img - 0.5) / 0.5
        img = fn.transpose(img, perm=[2, 0, 1])  # HWC → CHW
        pipe.set_outputs(img, labels)
    pipe.build()
    return pipe


def main():
    torch.cuda.set_device(0)
    device = 'gpu' if (len(sys.argv) > 1 and sys.argv[1] == 'gpu') else 'cpu'
    print(f"DALI file_reader 直读 {DATA}, decode device={device}")
    pipe = build_pipeline(device=device)

    from opt_eval.tensorrt.int8_ptq_test import EngineRunner
    runner = EngineRunner(ENGINE, batch_size=ENGINE_BATCH)

    def to_torch(out):
        return torch.as_tensor(out.as_tensor())

    # label 一致性: DALI label (目录排序编号) vs FastImageFolderDataset 规则
    out, labels = pipe.run()
    labels_np = labels.as_cpu().as_array()
    classes = sorted(d for d in os.listdir(DATA)
                     if os.path.isdir(os.path.join(DATA, d)))
    print(f"首批 label 范围: [{labels_np.min()}, {labels_np.max()}], "
          f"目录数 {len(classes)} (DALI label 应为目录排序编号)")

    # 预热 (首转保存第一批用于数值对拍)
    first_batch = None
    for i in range(3):
        out, _ = pipe.run()
        x = to_torch(out)
        if i == 0:
            first_batch = x.clone()
        xc = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        runner(xc[:ENGINE_BATCH])
    torch.cuda.synchronize()

    # 数值一致性抽查: 预热期第一批 (即首文件所在批) vs cv2 解码
    import cv2
    first_file = None
    for c in classes[:1]:
        for f in sorted(os.listdir(os.path.join(DATA, c)))[:1]:
            first_file = os.path.join(DATA, c, f)
    raw = open(first_file, 'rb').read()
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref = (img - 0.5) / 0.5
    x0 = first_batch[:1].cpu().numpy()
    print(f"数值一致性 (DALI vs cv2, 首文件): max_diff={np.abs(x0[0] - ref.transpose(2,0,1)).max():.6f}")

    # 吞吐: 跑完整 epoch
    t0 = time.time()
    cnt = 0
    while True:
        try:
            out, _ = pipe.run()
        except StopIteration:
            break
        except RuntimeError as e:
            if 'end of' in str(e).lower() or 'no more' in str(e).lower():
                break
            raise
        x = to_torch(out)
        b = x.shape[0]
        xc = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        # 尾批补齐逻辑: runner 分块执行
        for s in range(0, b, ENGINE_BATCH):
            e = min(s + ENGINE_BATCH, b)
            runner(xc[s:e])
        cnt += b
        if cnt >= 200000:
            break
    dt = time.time() - t0
    print(f"DALI({device}) 端到端: {cnt:,} 张 / {dt:.1f}s = {cnt/dt:,.0f} img/s "
          f"(cv2 链基线 ~3,644)")


if __name__ == '__main__':
    main()

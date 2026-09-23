"""模拟训练 epoch 末的外挂评估调用 (external_eval_backend=trt 分支端到端验证)。

直接 python 运行 (单进程 fabric), run_external_torch_eval 会以子进程拉起
eval_all_trt_single (8 卡, engine 缓存), 回读 JSON 结果。
"""
import os
import sys
from types import SimpleNamespace

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)

from lightning.fabric import Fabric  # noqa: E402

from external_torch_eval import run_external_torch_eval  # noqa: E402


def main():
    if int(os.environ.get('LOCAL_RANK', -1)) == -1:
        fabric = Fabric(devices=1)
        fabric.launch()
    else:
        fabric = Fabric(devices=1)

    cfg = SimpleNamespace(
        trainers=SimpleNamespace(
            external_eval_timeout_minutes=30,
            external_eval_fabric_bin='/root/anaconda3/envs/cvlface/bin/fabric',
            external_eval_backend='trt',
            external_eval_precision='fp16',
            num_gpu=8,
            output_dir='/tmp/fake_train_output_dir',
            task='bench_260922',
        ),
        evaluations=SimpleNamespace(
            yaml_path=os.path.join(WORK, 'evaluations/configs/bench_260922_1201.yaml'),
        ),
    )
    result = run_external_torch_eval(
        fabric=fabric, cfg=cfg,
        checkpoint_dir='/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/'
                       'adaface_ir101_webface12m',
        epoch=0)
    print('\n=== 训练侧拿到的 all_result ===')
    for k, v in result.items():
        print(f'{k}: {v}')
    print('EXTERNAL_TRT_EVAL_OK')


if __name__ == '__main__':
    main()

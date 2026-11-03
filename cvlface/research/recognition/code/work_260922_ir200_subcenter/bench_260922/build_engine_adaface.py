"""导出 adaface_ir101_webface12m → fp16 ONNX → TRT engine (供 portable_eval 使用).

计时项: 模型加载 / ONNX 导出 / engine 构建, 供启动开销对比。
"""
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os
import sys
import time

import torch

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.join(WORK_DIR, 'portable_eval'))
import eval_lib as el  # noqa: E402
from models import get_model  # noqa: E402
from general_utils.config_utils import load_config  # noqa: E402

MODEL_DIR = '/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engines')
BS = 512


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    onnx_path = os.path.join(OUT_DIR, 'adaface_ir101_bs512_fp16.onnx')
    engine_path = os.path.join(OUT_DIR, 'adaface_ir101_bs512_fp16.engine')

    torch.cuda.set_device(0)
    t0 = time.time()
    model_config = load_config(os.path.join(MODEL_DIR, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, 'work_0605')
    model.load_state_dict_from_path(os.path.join(MODEL_DIR, 'model.pt'))
    model.eval()
    t_load = time.time() - t0
    print(f'[计时] 模型加载: {t_load:.1f}s')

    t0 = time.time()
    export_model = model.half().cuda()
    dummy = torch.randn(BS, 3, 112, 112, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(export_model, dummy, onnx_path,
                          input_names=['input'], output_names=['output'],
                          opset_version=17, dynamo=False)
    torch.cuda.synchronize()
    t_export = time.time() - t0
    print(f'[计时] ONNX 导出 (fp16, bs={BS}): {t_export:.1f}s '
          f'({os.path.getsize(onnx_path) / 1e6:.0f} MB)')

    t0 = time.time()
    el.build_engine_from_onnx(onnx_path, engine_path, batch_size=BS, fp16=True)
    t_build = time.time() - t0
    print(f'[计时] TRT engine 构建: {t_build:.1f}s '
          f'({os.path.getsize(engine_path) / 1e6:.0f} MB)')

    # 快速验证: 同输入 torch vs TRT cosine
    x = torch.randn(BS, 3, 112, 112, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        ref = export_model(x).float()
        ref = torch.nn.functional.normalize(ref, dim=1)
    del export_model
    torch.cuda.empty_cache()
    eng = el.TrtEngine(engine_path, BS, device=0)
    out = eng.infer(x)
    out = torch.nn.functional.normalize(out, dim=1)
    cos = (ref * out).sum(1)
    print(f'[验证] torch(fp16) vs TRT cosine: mean={cos.mean():.5f} min={cos.min():.5f}')
    print(f'engine: {engine_path}')


if __name__ == '__main__':
    main()

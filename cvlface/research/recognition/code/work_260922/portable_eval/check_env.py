#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""迁移环境自检: 逐项检查依赖/硬件/数据, 缺什么给什么提示.

用法:
  python check_env.py                 # 基础检查
  python check_env.py --compile-test  # 额外实测 nvjpeg 扩展编译+解码
  python check_env.py --engine /path/engine --data /path/folder_or_rec
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

OK, BAD, WARN = '✓', '✗', '!'
ITEMS = []


def item(name, hint=''):
    def deco(fn):
        ITEMS.append((name, fn, hint))
        return fn
    return deco


def check_fn(fn, *args, **kw):
    try:
        return True, fn(*args, **kw)
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


# ---------- 基础 ----------

@item('python >= 3.10', '升级 Python')
def _():
    v = sys.version_info
    assert v >= (3, 10), f'当前 {v}'
    return f'{v.major}.{v.minor}.{v.micro}'


@item('torch + CUDA', 'pip install torch --index-url https://download.pytorch.org/whl/cu126')
def _():
    import torch
    assert torch.cuda.is_available(), 'torch.cuda 不可用 (驱动/CUDA 版本?)'
    return f'torch {torch.__version__}, cuda {torch.version.cuda}'


@item('GPU 数量', 'nvidia-smi 检查驱动')
def _():
    import torch
    n = torch.cuda.device_count()
    assert n >= 1, '未检测到 GPU'
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    return f'{n} × {names[0]}' + (' (多卡)' if n > 1 else '')


@item('tensorrt', 'pip install tensorrt (需与 CUDA 大版本匹配)')
def _():
    import tensorrt as trt
    return trt.__version__


@item('opencv (cv2)', 'pip install opencv-python-headless')
def _():
    import cv2
    return cv2.__version__


@item('numpy / tqdm', 'pip install numpy tqdm')
def _():
    import numpy, tqdm  # noqa
    return f'numpy {numpy.__version__}'


@item('ninja (扩展编译)', 'pip install ninja (并确保其在 PATH)')
def _():
    import ninja  # noqa
    # 绝对路径调 python (未 activate) 时同级 bin 目录可能不在 PATH, 自愈后再查
    exe_dir = str(Path(sys.executable).parent)
    if exe_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')
    p = shutil.which('ninja')
    assert p, 'python 包已装但 PATH 里无 ninja 可执行文件'
    return p


@item('CUDA 头文件 (cuda_runtime.h)', '装 CUDA toolkit 或 export CUDA_HOME=/usr/local/cuda')
def _():
    import eval_lib
    return eval_lib.find_cuda_include()


@item('nvjpeg 头文件+库 (GPU 解码必需, 仅 nvjpeg 管线)', 'pip install nvidia-nvjpeg-cu12 或 export NVJPEG_HOME=...')
def _():
    import eval_lib
    inc, lib = eval_lib.find_nvjpeg()
    return f'{inc} | {lib}'


@item('polars (可选, 仅生成 parquet 报表用)', 'pip install polars')
def _():
    import polars
    return polars.__version__


# ---------- 可选: 实测扩展编译与解码 ----------

@item('nvjpeg 扩展编译+解码实测 (--compile-test)', '看上方 nvjpeg/CUDA 头文件两项')
def _():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_lib
    import torch
    ext = eval_lib.build_nvjpeg_ext()
    ext.pe_nvjpeg_init(0, 8)
    out = torch.zeros(8, 112, 112, 3, dtype=torch.uint8, device='cuda')
    # 1×1 像素的合法 JPEG (灰色)
    import io
    import cv2
    _, enc = cv2.imencode('.jpg', np.full((112, 112, 3), 200, np.uint8))
    blob = enc.tobytes()
    host = torch.from_numpy(np.frombuffer(blob * 8, dtype=np.uint8).copy())
    offs = torch.arange(8, dtype=torch.int64) * len(blob)
    sizes = torch.full((8,), len(blob), dtype=torch.int64)
    ext.pe_decode_host(host, offs, sizes, out, 0,
                       torch.cuda.current_stream().cuda_stream, 1)
    assert out.unique().numel() <= 8, '解码输出异常'
    return '编译 + 解码通过'


@item('融合核 (fused_he_pn 快速匹配, --compile-test 实测编译)',
      '需 CUDA toolkit + ninja; 不可用时 --matcher auto 自动回退 torch 实现')
def _():
    p = Path(__file__).resolve().parent / 'cuda_histpn_he_fused.cu'
    if '--compile-test' not in sys.argv:
        assert p.exists(), f'{p.name} 缺失 (融合核随包分发, 勿删)'
        return '源码在位 (加 --compile-test 实测编译)'
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_lib
    ext = eval_lib._get_fused_ext()
    assert ext is not None, '融合核编译失败 (见上方输出)'
    return '编译通过'


@item('TRT engine 可用 (--engine 指定)', '用 build_engine_from_onnx 或训练侧导出')
def _():
    args = sys.argv
    if '--engine' not in args:
        return '未指定 (跳过)'
    p = Path(args[args.index('--engine') + 1])
    assert p.exists(), f'{p} 不存在'
    import eval_lib
    eng = eval_lib.TrtEngine(str(p), batch_size=8)
    return f'加载成功, 输出维度 {eng.output_dim}'


@item('数据源 (--data 指定 folder 或 rec 目录)', '先跑 make_synth_data.py 造测试数据')
def _():
    args = sys.argv
    if '--data' not in args:
        return '未指定 (跳过)'
    p = Path(args[args.index('--data') + 1])
    if (p / 'train.rec').exists():
        n = sum(1 for _ in open(p / 'train.idx'))
        return f'rec: {n} 条'
    n_dirs = sum(1 for d in p.iterdir() if d.is_dir())
    return f'folder: {n_dirs} 档案目录'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compile-test', action='store_true',
                    help='实测 nvjpeg 扩展编译与解码 (推荐迁移后首跑)')
    ap.add_argument('--engine', default='')
    ap.add_argument('--data', default='')
    args = ap.parse_args()

    print(f'环境自检: {Path(__file__).resolve().parent}\n' + '=' * 64)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    n_ok = n_bad = 0
    for name, fn, hint in ITEMS:
        if name.startswith('nvjpeg 扩展') and not args.compile_test:
            print(f'  {WARN} {name}  [跳过: 加 --compile-test 启用]')
            continue
        if (name.startswith('TRT engine') and not args.engine) or \
           (name.startswith('数据源') and not args.data):
            print(f'  {WARN} {name}  [跳过: 未指定]')
            continue
        ok, detail = check_fn(fn)
        if ok:
            print(f'  {OK} {name}: {detail}')
            n_ok += 1
        else:
            print(f'  {BAD} {name}: {detail}')
            if hint:
                print(f'      ↳ {hint}')
            n_bad += 1
    print('=' * 64)
    print(f'通过 {n_ok}, 失败 {n_bad}'
          + ('  →  按提示补齐后重跑' if n_bad else '  →  环境就绪'))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    import numpy as np  # noqa: F401  (compile-test 用)
    main()

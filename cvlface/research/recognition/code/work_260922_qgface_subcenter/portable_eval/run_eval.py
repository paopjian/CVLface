#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测主入口: 数据源 → (双流 nvjpeg / cv2) 提特征 → NxN 匹配 → TPIR@FPIR.

用法示例:
  # 1. 自造数据全链路自检 (推荐迁移后首跑)
  python make_synth_data.py --out /tmp/pe_data
  python run_eval.py --engine /path/engine_fp16 \
      --data /tmp/pe_data/rec_tiny --source rec \
      --pipeline nvjpeg --gpus 1 --self-test

  # 2. 真实测试集 (folder)
  python run_eval.py --engine /path/engine_fp16 \
      --data /data1/dataset_0918/test/test_34t --source folder \
      --pipeline nvjpeg --gpus 7 --tag mymodel_34t

  # 3. rec 数据
  python run_eval.py --engine ... --data /path/rec_dir --source rec ...

  # 4. 只建 engine (从 ONNX)
  python run_eval.py --build-from-onnx model.onnx --engine-out my.engine --bs 512

提特征结果自动缓存 (--no-cache 关闭), 同参重跑秒级返回。
"""
import argparse
import json
import sys
import time

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_lib as el  # noqa: E402


def do_build(args):
    assert args.build_from_onnx and args.engine_out, \
        '--build-from-onnx 需配合 --engine-out'
    p = el.build_engine_from_onnx(args.build_from_onnx, args.engine_out,
                                  batch_size=args.bs, fp16=not args.fp32,
                                  dynamic_max=args.dynamic_max)
    print(f'engine 已构建: {p}')
    return 0


def do_extract_and_eval(args):
    engine_path = args.engine
    assert engine_path and Path(engine_path).exists(), \
        f'engine 不存在: {engine_path} (或用 --build-from-onnx 先构建)'

    tag = args.tag or (Path(args.data).name + f'_{args.pipeline}')
    t0 = time.time()

    # ---- 1. 提特征 (带缓存) ----
    cache_ok = not args.no_cache and \
        (Path(args.out_dir) / f'{tag}.feats.fp16.npy').exists()
    if cache_ok:
        feats, ids, meta = el.load_feature_cache(args.out_dir, tag)
        print(f'[特征缓存] {tag}: {feats.shape[0]:,} 图, 跳过提取')
    else:
        feats, ids, meta = el.extract_features(
            args.source, args.data, engine_path, args.out_dir,
            num_gpus=args.gpus, batch_size=args.bs, pipeline=args.pipeline,
            decode_threads=args.decode_threads,
            nvjpeg_chunk=args.nvjpeg_chunk)
        print(f"[提取] {meta['n_images']:,} 图, {meta['extract_time_s']}s "
              f"({meta['throughput_img_s']:,} img/s 聚合)")
        if not args.no_cache:
            el.save_feature_cache(args.out_dir, tag, feats, ids, meta)
    total_time = time.time() - t0

    # ---- 2. 自检 (自造数据用) ----
    if args.self_test:
        if args.source == 'rec':
            src = el.RecSource(args.data)
        else:
            src = el.FolderSource(args.data)
        assert meta['n_images'] == len(src), \
            f'闭合失败: 提取 {meta["n_images"]} != 扫描 {len(src)}'
        assert len(np.unique(ids)) == len(src.id_map if hasattr(src, 'id_map')
                                         else np.unique(ids)), '档案数异常'
        print(f'[自检] 提取数/扫描数闭合 ✓, 档案数 {len(np.unique(ids)):,} ✓')

    # ---- 3. NxN 匹配 + TPIR@FPIR ----
    t0 = time.time()
    if args.matcher == 'torch':
        hist_fn = el.pos_neg_hist_torch
    else:
        if args.matcher == 'fused' and el._get_fused_ext() is None:
            raise RuntimeError('融合核不可用 (编译失败, 见上方输出); '
                               '用 --matcher torch 或 auto')
        hist_fn = el.pos_neg_hist_fused  # auto/fused: 内部失败自动回退 torch
    block = args.block or (8192 if args.matcher == 'torch' else 16384)
    pos_hist, neg_hist = hist_fn(feats, ids, num_gpus=args.gpus,
                                 block=block, bins=args.bins)
    match_time = time.time() - t0
    results, thresholds, (total_pos, total_neg) = el.tpir_from_hist(
        pos_hist, neg_hist, args.bins, fpirs=el.DEFAULT_FPIRS)

    print(f'\n[匹配] {meta["n_images"]:,} 图: 正对 {total_pos:,} / '
          f'负对 {total_neg:,}, 耗时 {match_time:.1f}s')
    print(f"{'FPIR':>10} {'TPIR %':>10} {'阈值':>10}")
    for k in results:
        print(f"{k:>10} {results[k]:>10.3f} {thresholds[k]:>10.4f}")

    out_json = Path(args.out_dir) / f'eval_{tag}.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'tag': tag, 'meta': meta,
                   'match_time_s': round(match_time, 1),
                   'total_time_s': round(total_time, 1),
                   'pos_pairs': int(total_pos), 'neg_pairs': int(total_neg),
                   'tpir': results, 'thresholds': thresholds,
                   'bins': args.bins, 'block': block,
                   'matcher': (hist_fn.__name__.rsplit('_', 1)[-1])}, f,
                  ensure_ascii=False, indent=2)
    print(f'结果已保存: {out_json}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', default='', help='TRT engine 路径 (fp16 静态整批)')
    ap.add_argument('--build-from-onnx', default='', help='从 ONNX 构建 engine 后退出')
    ap.add_argument('--engine-out', default='', help='构建 engine 的输出路径')
    ap.add_argument('--fp32', action='store_true', help='构建时不用 fp16')
    ap.add_argument('--data', default='', help='folder 根目录 或 rec 目录 (含 train.rec)')
    ap.add_argument('--source', default='folder', choices=['folder', 'rec'])
    ap.add_argument('--pipeline', default='nvjpeg', choices=['nvjpeg', 'cv2'],
                    help='nvjpeg=GPU 批量解码+双流(推荐) / cv2=CPU 解码兜底')
    ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--bs', type=int, default=512, help='engine 静态 batch '
                    '(动态构建时为 opt 点与推理缓冲批容量)')
    ap.add_argument('--dynamic-max', type=int, default=0,
                    help='>0 时构建动态 batch engine (max=该值, min=64, opt=--bs); '
                    '静态构建时 engine batch 由 ONNX 声明决定')
    ap.add_argument('--decode-threads', type=int, default=4)
    ap.add_argument('--nvjpeg-chunk', type=int, default=256,
                    help='nvjpegDecodeBatched 单次调用张数 (512×大JPEG 会段错误)')
    ap.add_argument('--bins', type=int, default=2000)
    ap.add_argument('--block', type=int, default=0,
                    help='NxN 匹配分块, 0=按匹配器取默认 (torch 8192 / '
                         'fused 16384); 越大越快越吃显存')
    ap.add_argument('--matcher', default='auto', choices=['auto', 'fused', 'torch'],
                    help='NxN 匹配实现: auto=融合核优先失败回退 torch (默认) / '
                         'fused=v7 融合核 (强制) / torch=纯 torch')
    ap.add_argument('--tag', default='', help='输出文件标签 (默认 数据目录名_管线)')
    ap.add_argument('--out-dir', default='./results')
    ap.add_argument('--no-cache', action='store_true', help='强制重新提特征')
    ap.add_argument('--self-test', action='store_true',
                    help='自造数据场景: 校验提取数/扫描数闭合与档案数')
    args = ap.parse_args()

    if args.build_from_onnx:
        sys.exit(do_build(args))
    if not args.data:
        ap.print_help()
        sys.exit(2)
    sys.exit(do_extract_and_eval(args))


if __name__ == '__main__':
    main()

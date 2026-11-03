"""
评估方法对比 benchmark 脚本

对比 5 种配置在 val_20260605 上的耗时:
  1. 无 compile, FP32
  2. 无 compile, BF16
  3. compile max-autotune, BF16
  4. TRT FP16

每个评估器分别计时: 特征提取 / 评估计算 / 总耗时
最终输出对比表格和 JSON 结果。

用法:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python benchmark_eval_methods.py \
    --ckpt_path /data2/dataset_0605/train_output/s2_body36_0605_06-10_2/checkpoints_every_epoch/epoch:0_step:9053 \
    --num_gpu 7 \
    --eval_config_name val_20260605

单卡:
  python benchmark_eval_methods.py \
    --ckpt_path <path> --num_gpu 1
"""
import os
import sys
import json
import time
import signal
import subprocess
import argparse

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_torch_eval(ckpt_path, num_gpu, precision, eval_config_name, compile_mode=None, timeout_min=60):
    """运行 eval_all_torch_single.py 并返回 timing 结果"""
    script = os.path.join(SCRIPT_DIR, 'eval_all_torch_single.py')

    if num_gpu > 1:
        cmd = [
            'fabric', 'run',
            f'--strategy=ddp',
            f'--devices={num_gpu}',
            f'--precision={precision}',
            script,
            '--eval_config_name', eval_config_name,
            '--single_ckpt_path', ckpt_path,
            '--num_gpu', str(num_gpu),
            '--precision', precision,
            '--name', 'benchmark_tmp',
            '--timing',
        ]
    else:
        cmd = [
            PYTHON, script,
            '--eval_config_name', eval_config_name,
            '--single_ckpt_path', ckpt_path,
            '--num_gpu', '1',
            '--precision', precision,
            '--name', 'benchmark_tmp',
            '--timing',
        ]

    if compile_mode:
        cmd.extend(['--compile', '--compile_mode', compile_mode])

    print(f"  CMD: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        stdout, _ = proc.communicate(timeout=timeout_min * 60)
        return proc.returncode, stdout
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        return -1, "TIMEOUT"


def run_trt_eval(ckpt_path, num_gpu, eval_config_name, timeout_min=60):
    """运行 eval_all_trt_single.py 并返回 timing 结果"""
    script = os.path.join(SCRIPT_DIR, 'eval_all_trt_single.py')
    cmd = [
        PYTHON, script,
        '--eval_config_name', eval_config_name,
        '--ckpt_path', ckpt_path,
        '--num_gpu', str(num_gpu),
        '--name', 'benchmark_trt_tmp',
    ]

    print(f"  CMD: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        stdout, _ = proc.communicate(timeout=timeout_min * 60)
        return proc.returncode, stdout
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        return -1, "TIMEOUT"


def parse_timing_torch(stdout):
    """从 eval_all_torch_single 的 stdout 解析计时结果"""
    timings = {}
    in_summary = False
    for line in stdout.split('\n'):
        line = line.strip()
        # 解析计时汇总块
        if '计时汇总' in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith('总耗时:'):
                parts = line.split()
                timings['total_sec'] = float(parts[1].replace('s', ''))
                in_summary = False
            elif ':' in line and 's' in line and not line.startswith('=') and not line.startswith('─'):
                # "  evaluator_name: 123.45s (2.06min)"
                parts = line.split(':')
                if len(parts) == 2:
                    name = parts[0].strip()
                    val_str = parts[1].strip().split('s')[0]
                    try:
                        timings[name] = float(val_str)
                    except ValueError:
                        pass
    return timings


def parse_timing_trt(stdout):
    """从 eval_all_trt_single 的 stdout 解析计时结果"""
    timings = {}
    current_eval = None
    for line in stdout.split('\n'):
        line = line.strip()
        # "评估: work_0605_3t (type=custom_verification4)"
        if line.startswith('评估:') and '(' in line:
            current_eval = line.split('评估:')[1].split('(')[0].strip()
        # "  特征提取: 112.6s (7 GPU)"
        if '特征提取:' in line and current_eval:
            try:
                val = float(line.split('特征提取:')[1].split('s')[0].strip())
                timings[f'{current_eval}_extract'] = val
            except (ValueError, IndexError):
                pass
        # "  指标计算: 40.2s"
        if '指标计算:' in line and current_eval:
            try:
                val = float(line.split('指标计算:')[1].split('s')[0].strip())
                timings[f'{current_eval}_metric'] = val
            except (ValueError, IndexError):
                pass
        # "TRT engine 构建: 56.3s"
        if 'TRT engine 构建:' in line:
            try:
                val = float(line.split('构建:')[1].split('s')[0].strip())
                timings['trt_build'] = val
            except (ValueError, IndexError):
                pass
    return timings


def print_comparison_table(results):
    """打印对比表格"""
    configs = list(results.keys())
    # 收集所有评估器名
    all_evals = set()
    for cfg_data in results.values():
        for key in cfg_data.get('timings', {}):
            if key not in ('total_sec', 'combined_evaluations', 'trt_build'):
                # torch 格式: evaluator_name
                # trt 格式: evaluator_name_extract / evaluator_name_metric
                base_name = key.replace('_extract', '').replace('_metric', '')
                all_evals.add(base_name)

    print("\n" + "=" * 80)
    print("评估方法对比结果")
    print("=" * 80)

    # Header
    header = f"{'评估器':<25}"
    for cfg in configs:
        header += f"| {cfg:<18}"
    print(header)
    print("-" * len(header))

    # 每个评估器一行
    sorted_evals = sorted(all_evals)
    for ev in sorted_evals:
        row = f"{ev:<25}"
        for cfg in configs:
            timings = results[cfg].get('timings', )
            # torch 格式
            if ev in timings:
                row += f"| {timings[ev]:>8.1f}s{'':>9}"
            # trt 格式
            elif f'{ev}_extract' in timings:
                ext = timings.get(f'{ev}_extract', 0)
                met = timings.get(f'{ev}_metric', 0)
                row += f"| {ext+met:>8.1f}s{'':>9}"
            else:
                row += f"| {'N/A':>18}"
        print(row)

    # 总计行
    print("-" * len(header))
    row = f"{'总耗时':<25}"
    for cfg in configs:
        timings = results[cfg].get('timings', {})
        total = timings.get('total_sec', 0)
        if total == 0:
            # TRT: 累加
            total = sum(v for k, v in timings.items() if k not in ('trt_build',))
        row += f"| {total:>8.1f}s{'':>9}"
    print(row)

    # TRT build 时间
    row = f"{'准备耗时 (首次)':<25}"
    for cfg in configs:
        timings = results[cfg].get('timings', {})
        if 'trt_build' in timings:
            row += f"| {timings['trt_build']:>8.1f}s{'':>9}"
        else:
            row += f"| {'0':>18}"
    print(row)
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估方法对比 benchmark')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Checkpoint 路径')
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--eval_config_name', type=str, default='val_20260605')
    parser.add_argument('--timeout_minutes', type=int, default=60)
    parser.add_argument('--skip', nargs='*', default=[], choices=['fp32', 'bf16', 'compile', 'trt'],
                        help='跳过指定配置')
    args = parser.parse_args()

    print(f"Checkpoint: {args.ckpt_path}")
    print(f"GPU 数量: {args.num_gpu}")
    print(f"评估配置: {args.eval_config_name}")
    print(f"超时: {args.timeout_minutes} min")
    print()

    results = {}

    # === 1. 无 compile, FP32 ===
    if 'fp32' not in args.skip:
        print(f"\n{'='*60}")
        print("[1/4] 无 compile, FP32 (32-true)")
        print(f"{'='*60}")
        t0 = time.time()
        rc, stdout = run_torch_eval(
            args.ckpt_path, args.num_gpu, '32-true', args.eval_config_name,
            compile_mode=None, timeout_min=args.timeout_minutes)
        wall_time = time.time() - t0
        if rc == 0:
            timings = parse_timing_torch(stdout)
            timings['wall_time'] = wall_time
            results['FP32 无compile'] = {'timings': timings, 'returncode': rc}
            print(f"  完成, wall time: {wall_time:.1f}s")
        else:
            print(f"  失败 (rc={rc})")
            if stdout != "TIMEOUT":
                # 打印最后 20 行
                lines = stdout.strip().split('\n')
                for l in lines[-20:]:
                    print(f"    {l}")
            results['FP32 无compile'] = {'timings': {}, 'returncode': rc, 'error': stdout[-500:] if stdout else ''}

    # === 2. 无 compile, BF16 ===
    if 'bf16' not in args.skip:
        print(f"\n{'='*60}")
        print("[2/4] 无 compile, BF16 (bf16-mixed)")
        print(f"{'='*60}")
        t0 = time.time()
        rc, stdout = run_torch_eval(
            args.ckpt_path, args.num_gpu, 'bf16-mixed', args.eval_config_name,
            compile_mode=None, timeout_min=args.timeout_minutes)
        wall_time = time.time() - t0
        if rc == 0:
            timings = parse_timing_torch(stdout)
            timings['wall_time'] = wall_time
            results['BF16 无compile'] = {'timings': timings, 'returncode': rc}
            print(f"  完成, wall time: {wall_time:.1f}s")
        else:
            print(f"  失败 (rc={rc})")
            if stdout != "TIMEOUT":
                lines = stdout.strip().split('\n')
                for l in lines[-20:]:
                    print(f"    {l}")
            results['BF16 无compile'] = {'timings': {}, 'returncode': rc, 'error': stdout[-500:] if stdout else ''}

    # === 3. compile max-autotune, BF16 ===
    if 'compile' not in args.skip:
        print(f"\n{'='*60}")
        print("[3/4] compile max-autotune, BF16 (bf16-mixed)")
        print(f"{'='*60}")
        t0 = time.time()
        rc, stdout = run_torch_eval(
            args.ckpt_path, args.num_gpu, 'bf16-mixed', args.eval_config_name,
            compile_mode='max-autotune', timeout_min=args.timeout_minutes)
        wall_time = time.time() - t0
        if rc == 0:
            timings = parse_timing_torch(stdout)
            timings['wall_time'] = wall_time
            results['compile BF16'] = {'timings': timings, 'returncode': rc}
            print(f"  完成, wall time: {wall_time:.1f}s")
        else:
            print(f"  失败 (rc={rc})")
            if stdout != "TIMEOUT":
                lines = stdout.strip().split('\n')
                for l in lines[-20:]:
                    print(f"    {l}")
            results['compile BF16'] = {'timings': {}, 'returncode': rc, 'error': stdout[-500:] if stdout else ''}

    # === 4. TensorRT FP16 ===
    if 'trt' not in args.skip:
        print(f"\n{'='*60}")
        print("[4/4] TensorRT FP16")
        print(f"{'='*60}")
        t0 = time.time()
        rc, stdout = run_trt_eval(
            args.ckpt_path, args.num_gpu, args.eval_config_name,
            timeout_min=args.timeout_minutes)
        wall_time = time.time() - t0
        if rc == 0:
            timings = parse_timing_trt(stdout)
            timings['wall_time'] = wall_time
            results['TRT FP16'] = {'timings': timings, 'returncode': rc}
            print(f"  完成, wall time: {wall_time:.1f}s")
        else:
            print(f"  失败 (rc={rc})")
            if stdout != "TIMEOUT":
                lines = stdout.strip().split('\n')
                for l in lines[-20:]:
                    print(f"    {l}")
            results['TRT FP16'] = {'timings': {}, 'returncode': rc, 'error': stdout[-500:] if stdout else ''}

    # === 输出结果 ===
    print_comparison_table(results)

    # 保存 JSON
    output_path = os.path.join(SCRIPT_DIR, 'benchmark_eval_methods_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存: {output_path}")

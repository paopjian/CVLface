"""Run isolated TinyFace external evaluations several times.

Each run starts a fresh 8-process Fabric job, matching the training hook in
``external_torch_eval.py``.  By default the child environment is untouched;
``--threads`` optionally pins TINYFACE/OMP/MKL thread counts for comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='checkpoint directory containing model.pt')
    parser.add_argument('--runs', type=int, default=2, help='number of fresh sequential runs')
    parser.add_argument('--threads', type=int, default=None,
                        help='optional TINYFACE/OMP/MKL thread count; omit to preserve training environment')
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--precision', default='bf16-mixed')
    parser.add_argument('--eval_config_name', default='test_20260605_tinyface')
    parser.add_argument('--timeout_minutes', type=int, default=90)
    parser.add_argument('--no_compile', action='store_true', help='disable torch.compile')
    parser.add_argument('--fabric_bin', default='', help='fabric executable; defaults to the current Python environment')
    parser.add_argument('--output_dir', default='', help='directory for logs and result JSON files')
    return parser.parse_args()


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be positive')
    if args.threads is not None and args.threads < 1:
        raise ValueError('--threads must be positive')

    script_dir = Path(__file__).resolve().parents[2]
    single_eval = script_dir / 'eval_all_torch_single.py'
    fabric_bin = args.fabric_bin or str(Path(sys.executable).with_name('fabric'))
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / 'benchmark_results' / 'tinyface_external_restarts'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for run_index in range(1, args.runs + 1):
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        stem = f'run_{run_index:02d}_{stamp}'
        result_path = output_dir / f'{stem}.json'
        log_path = output_dir / f'{stem}.log'
        command = [
            fabric_bin, 'run',
            f'--devices={args.num_gpu}',
            f'--precision={args.precision}',
            str(single_eval),
            '--eval_config_name', args.eval_config_name,
            '--pipeline_name', 'default',
            '--single_ckpt_path', args.checkpoint,
            '--name', f'tinyface_restart_{run_index}',
            '--project_name', 'tinyface_external_restart_test',
            '--num_gpu', str(args.num_gpu),
            '--precision', args.precision,
            '--timeout_minutes', str(args.timeout_minutes),
            '--result_path', str(result_path),
            '--timing',
        ]
        if not args.no_compile:
            command.append('--compile')

        child_env = os.environ.copy()
        conda_prefix = child_env.get('CONDA_PREFIX')
        if conda_prefix and not child_env.get('LD_LIBRARY_PATH'):
            child_env['LD_LIBRARY_PATH'] = str(Path(conda_prefix) / 'lib')
        if args.threads is not None:
            thread_value = str(args.threads)
            child_env['TINYFACE_NUM_THREADS'] = thread_value
            child_env['OMP_NUM_THREADS'] = thread_value
            child_env['MKL_NUM_THREADS'] = thread_value
        child_env['WANDB_MODE'] = 'disabled'
        child_env['PYTHONUNBUFFERED'] = '1'

        print(f'[{run_index}/{args.runs}] starting fresh external evaluation')
        print('  checkpoint:', args.checkpoint)
        print('  threads:', args.threads if args.threads is not None else 'preserve environment')
        print('  log:', log_path)
        start = time.monotonic()
        with log_path.open('w', encoding='utf-8') as log_handle:
            log_handle.write('$ ' + ' '.join(command) + '\n')
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=script_dir,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=args.timeout_minutes * 60)
            except subprocess.TimeoutExpired:
                stop_process_group(process)
                return_code = -9
        wall_seconds = time.monotonic() - start

        record = {
            'run': run_index,
            'threads': args.threads,
            'return_code': return_code,
            'wall_seconds': wall_seconds,
            'log_path': str(log_path),
            'result_path': str(result_path),
        }
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding='utf-8'))
                record['reported_total_seconds'] = payload.get('total_elapsed_seconds')
                record['timing_results'] = payload.get('timing_results', {})
            except json.JSONDecodeError as error:
                record['result_error'] = str(error)
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        if return_code != 0:
            print(f'run {run_index} failed; stopping remaining runs', file=sys.stderr)
            break

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('summary:', summary_path)
    return 0 if summary and all(item['return_code'] == 0 for item in summary) else 1


if __name__ == '__main__':
    raise SystemExit(main())

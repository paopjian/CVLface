"""完整空转训练 RecordIO，并在每个 epoch 后运行 TinyFace 外部评估。

该脚本保留真实训练 DataLoader 的数据访问、解码、增强、worker 数量和
persistent_workers 行为，但不构造模型、不执行 forward/backward，也不保存
训练 checkpoint。用于区分 IO/worker 内存压力与 GPU 训练计算的影响。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__, indicator=["__root__.txt"], pythonpath=True, dotenv=True
)
CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import omegaconf
import torch
from tqdm import tqdm
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

from dataset.augment_dataset_v2 import AugmentMXDatasetV2
from fabric.fabric import setup_dataloader_from_dataset


_DISTRIBUTED_ENV_KEYS = (
    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
    'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
    'TORCHELASTIC_RUN_ID',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', default='/data1/dataset_0605')
    parser.add_argument('--rec', default='train_rec')
    parser.add_argument('--data_aug_config', default='data_augs/configs/gridsample_v2_numpy.yaml')
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--eval_config_name', default='test_20260605_tinyface')
    parser.add_argument('--precision', default='bf16-mixed')
    parser.add_argument('--timeout_minutes', type=int, default=90)
    parser.add_argument('--output_dir', default='/tmp/tinyface_empty_read_3epochs')
    parser.add_argument('--fabric_bin', default='/root/anaconda3/envs/cvlface/bin/fabric')
    return parser.parse_args()


def read_smaps_rollup(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(f'/proc/{pid}/smaps_rollup', encoding='utf-8') as handle:
            for line in handle:
                key, value, *_ = line.split()
                if key.rstrip(':') in {'Rss', 'Pss', 'Private_Dirty', 'Shared_Dirty'}:
                    values[key.rstrip(':')] = int(value) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return values


def stop_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temporary_path.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(temporary_path, path)


def run_external_eval(args: argparse.Namespace, fabric: Fabric, epoch: int, output_dir: Path) -> None:
    result_path = output_dir / 'external_eval' / f'epoch_{epoch}_raw.json'
    log_path = output_dir / 'external_eval' / f'epoch_{epoch}.log'
    status_path = output_dir / 'external_eval' / f'epoch_{epoch}_status.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if fabric.local_rank == 0:
        for stale_path in (result_path, status_path):
            stale_path.unlink(missing_ok=True)
    fabric.barrier()

    if fabric.local_rank == 0:
        eval_script = Path(__file__).resolve().parents[2] / 'eval_all_torch_single.py'
        main_port = find_available_port()
        command = [
            args.fabric_bin, 'run', f'--devices={args.num_gpu}',
            f'--precision={args.precision}', '--main-address=127.0.0.1',
            f'--main-port={main_port}', str(eval_script),
            '--eval_config_name', args.eval_config_name,
            '--pipeline_name', 'default',
            '--single_ckpt_path', args.checkpoint,
            '--name', f'empty_read_epoch_{epoch}',
            '--project_name', 'tinyface_empty_read_benchmark',
            '--num_gpu', str(args.num_gpu),
            '--precision', args.precision,
            '--timeout_minutes', str(args.timeout_minutes),
            '--result_path', str(result_path), '--timing', '--compile',
        ]
        child_env = os.environ.copy()
        for key in _DISTRIBUTED_ENV_KEYS:
            child_env.pop(key, None)
        child_env['WANDB_MODE'] = 'disabled'
        child_env['PYTHONUNBUFFERED'] = '1'
        child_env['TINYFACE_NUM_THREADS'] = '32'
        child_env['OMP_NUM_THREADS'] = '32'
        child_env['MKL_NUM_THREADS'] = '32'
        print('External evaluation:', ' '.join(command), flush=True)
        progress = None
        try:
            with log_path.open('w', encoding='utf-8') as log_handle:
                process = subprocess.Popen(
                    command, cwd=eval_script.parent, env=child_env,
                    stdout=log_handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=args.timeout_minutes * 60)
                except subprocess.TimeoutExpired:
                    stop_group(process)
                    raise RuntimeError(f'epoch {epoch} external evaluation timed out')
            if return_code != 0:
                raise RuntimeError(f'epoch {epoch} external evaluation failed: {return_code}')
            atomic_write_json(status_path, {'ok': True})
        except Exception as error:
            atomic_write_json(status_path, {'ok': False, 'error': str(error)})
    else:
        deadline = time.monotonic() + args.timeout_minutes * 60 + 120
        while not status_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'timed out waiting for {status_path}')
            time.sleep(1)

    fabric.barrier()
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if not status['ok']:
        raise RuntimeError(status['error'])


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = omegaconf.OmegaConf.load(args.data_aug_config)
    dataset = AugmentMXDatasetV2(
        root_dir=os.path.join(args.data_root, args.rec),
        local_rank=int(os.environ.get('LOCAL_RANK', 0)),
        augmentation_version=aug_cfg.augmentation_version,
        aug_params=aug_cfg.aug_params,
    )
    dataset.color_space = 'RGB'

    ddp_strategy = DDPStrategy(timeout=dt.timedelta(minutes=args.timeout_minutes))
    fabric = Fabric(
        precision=args.precision, accelerator='auto', strategy=ddp_strategy,
        devices=args.num_gpu,
    )
    if args.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = lambda **kwargs: setup_dataloader_from_dataset(
        seed=2048, fabric=fabric, **kwargs
    )
    dataloader = fabric.setup_dataloader_from_dataset(
        dataset=dataset, is_train=True, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if fabric.local_rank == 0:
        print(f'Empty-read dataset size: {len(dataset)}', flush=True)
        print(f'workers={args.num_workers}, batch_size={args.batch_size}, epochs={args.epochs}', flush=True)

    for epoch in range(args.epochs):
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        start = time.perf_counter()
        batch_count = 0
        progress_disabled = fabric.local_rank != 0
        try:
            progress = tqdm(
                dataloader, total=len(dataloader),
                desc=f'Empty read epoch {epoch}', unit='batch',
                disable=progress_disabled, file=sys.stderr,
            )
            for batch_count, batch in enumerate(progress, start=1):
                # Force the batch to be materialized and consumed, matching the
                # training DataLoader's decode/augment/pin-memory path.
                if isinstance(batch, (tuple, list)):
                    for value in batch:
                        if torch.is_tensor(value):
                            value.numel()
                elif isinstance(batch, dict):
                    for value in batch.values():
                        if torch.is_tensor(value):
                            value.numel()
        finally:
            if progress is not None:
                progress.close()
        elapsed = time.perf_counter() - start
        memory = read_smaps_rollup(os.getpid())
        if fabric.local_rank == 0:
            print(
                f'EMPTY epoch={epoch} batches={batch_count} elapsed={elapsed:.2f}s '
                f'RSS={memory.get("Rss", 0) / 2**30:.1f}GiB '
                f'PSS={memory.get("Pss", 0) / 2**30:.1f}GiB '
                f'PrivateDirty={memory.get("Private_Dirty", 0) / 2**30:.1f}GiB',
                flush=True,
            )
            (output_dir / f'empty_epoch_{epoch}.json').write_text(
                json.dumps({'epoch': epoch, 'batches': batch_count, 'elapsed_seconds': elapsed, 'memory': memory}, indent=2),
                encoding='utf-8',
            )
        fabric.barrier()
        run_external_eval(args, fabric, epoch, output_dir)

    if fabric.local_rank == 0:
        print(f'completed: {output_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

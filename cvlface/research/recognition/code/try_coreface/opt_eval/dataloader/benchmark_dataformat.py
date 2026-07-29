"""
Data format benchmark: compare 6 formats for face recognition training.

Usage (single GPU):
    python benchmark_dataformat.py --data_root /data1/dataset_0605/try

Usage (multi-GPU):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run \
        --strategy=ddp --devices=7 --precision="bf16-mixed" \
        benchmark_dataformat.py --data_root /data1/dataset_0605/try3
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
sys.path.append(os.path.join(root))

import argparse
import time
import json
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

try:
    from lightning.fabric import Fabric
except ImportError:
    Fabric = None


def get_transform():
    """Standard face recognition transform: 112x112, normalize."""
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_model(model_name='ir18'):
    """Get a model for benchmarking."""
    from models import get_model as _get_model
    from general_utils.config_utils import load_config

    if model_name == 'ir18':
        yaml_path = 'iresnet/configs/v1_ir18.yaml'
    else:
        yaml_path = 'iresnet/configs/v1_ir101.yaml'

    # Try loading from config
    try:
        model_cfg = load_config(yaml_path)
        model = _get_model(model_cfg, yaml_path)
    except Exception:
        # Fallback: simple CNN for benchmarking
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
        )
    return model


def get_dataset(fmt, data_root, benchmark_root, transform):
    """Create dataset for the given format."""
    from dataset.benchmark_datasets import (
        ImageFolderFaceDataset, LMDBFaceDataset, RecordIOFaceDataset,
        WebDatasetFaceDataset, HDF5FaceDataset, ZarrFaceDataset,
        StreamingFaceDataset
    )

    if fmt == 'imagefolder':
        return ImageFolderFaceDataset(data_root, transform=transform)
    elif fmt == 'lmdb':
        return LMDBFaceDataset(os.path.join(benchmark_root, 'lmdb'), transform=transform)
    elif fmt == 'recordio':
        return RecordIOFaceDataset(os.path.join(benchmark_root, 'recordio'), transform=transform)
    elif fmt == 'webdataset':
        return WebDatasetFaceDataset(os.path.join(benchmark_root, 'webdataset'), transform=transform)
    elif fmt == 'hdf5':
        return HDF5FaceDataset(os.path.join(benchmark_root, 'hdf5'), transform=transform)
    elif fmt == 'zarr':
        return ZarrFaceDataset(os.path.join(benchmark_root, 'zarr'), transform=transform)
    elif fmt == 'streaming':
        return StreamingFaceDataset(os.path.join(benchmark_root, 'streaming'), transform=transform)
    else:
        raise ValueError(f'Unknown format: {fmt}')


def get_disk_usage(path):
    """Get disk usage in MB."""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def benchmark_format(fmt, data_root, benchmark_root, model, transform,
                     batch_size, num_workers, fabric=None, num_classes=None):
    """Benchmark a single format: 1 epoch of forward+backward."""
    result = {'format': fmt}
    rank = fabric.global_rank if fabric else 0

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # Dataset
    try:
        log(f'  [1/5] Creating dataset...')
        t0 = time.time()
        dataset = get_dataset(fmt, data_root, benchmark_root, transform)
        result['dataset_load_time'] = time.time() - t0
        result['num_samples'] = len(dataset)
        # Auto-detect num_classes from ImageFolder dataset if not specified
        if num_classes is None:
            if hasattr(dataset, 'class_to_idx'):
                num_classes = len(dataset.class_to_idx)
            else:
                # Count subdirectories in data_root as fallback
                num_classes = len([d for d in os.listdir(data_root)
                                   if os.path.isdir(os.path.join(data_root, d))])
        log(f'  [1/5] Dataset ready: {len(dataset)} samples, {num_classes} classes ({time.time()-t0:.1f}s)')
    except Exception as e:
        result['error'] = f'Dataset creation failed: {e}'
        traceback.print_exc()
        return result

    # Disk usage
    if fmt == 'imagefolder':
        result['disk_mb'] = get_disk_usage(data_root)
    else:
        fmt_dir = os.path.join(benchmark_root, fmt)
        result['disk_mb'] = get_disk_usage(fmt_dir)

    # DataLoader
    log(f'  [2/5] Creating DataLoader (workers={num_workers})...')
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    # StreamingDataset handles distributed splitting internally — no DistributedSampler needed
    is_streaming = (fmt == 'streaming')

    # Zarr+tensorstore requires spawn (fork triggers abort due to internal threads)
    mp_context = 'spawn' if fmt == 'zarr' and num_workers > 0 else None

    if is_iterable:
        # WebDataset is IterableDataset — no sampler, no shuffle
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    elif is_streaming:
        # StreamingDataset already splits data per-rank, skip DistributedSampler
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True, drop_last=True,
                                persistent_workers=num_workers > 0)
    elif fabric and fabric.world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=fabric.world_size,
                                     rank=fabric.global_rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True, drop_last=True,
                                persistent_workers=num_workers > 0,
                                multiprocessing_context=mp_context)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True, drop_last=True,
                                persistent_workers=num_workers > 0,
                                multiprocessing_context=mp_context)

    log(f'  [2/5] DataLoader ready.')

    # Model + classifier head
    log(f'  [3/5] Setting up model + classifier...')
    model_copy = model
    classifier = nn.Linear(512, num_classes)

    if fabric:
        model_copy = fabric.setup(model_copy)
        classifier = fabric.setup(classifier)
        if is_streaming:
            # StreamingDataset handles rank-splitting internally; skip Fabric's sampler injection
            dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
        else:
            dataloader = fabric.setup_dataloaders(dataloader)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_copy = model_copy.to(device)
        classifier = classifier.to(device)

    model_copy.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    params = list(model_copy.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
    log(f'  [3/5] Model ready on device.')

    # Warmup: 5 batches
    data_times = []
    batch_times = []

    if fabric:
        device = fabric.device

    # Run 1 epoch
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    world_size = fabric.world_size if fabric else 1
    # StreamingDataset.len() already returns per-rank count, don't divide by world_size again
    if is_streaming:
        total_batches_est = len(dataset) // batch_size if hasattr(dataset, '__len__') and len(dataset) > 0 else '?'
    else:
        total_batches_est = len(dataset) // (batch_size * world_size) if hasattr(dataset, '__len__') and len(dataset) > 0 else '?'
    log(f'  [4/5] Training loop starting (~{total_batches_est} batches)...')

    epoch_start = time.time()
    num_batches = 0
    data_start = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        data_end = time.time()
        data_times.append(data_end - data_start)

        if not fabric:
            images = images.to(device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.long).to(device, non_blocking=True) \
                if not isinstance(labels, torch.Tensor) else labels.to(device, non_blocking=True)

        batch_start = time.time()

        # Forward
        features = model_copy(images)
        logits = classifier(features)
        loss = criterion(logits, labels)

        # Backward
        if fabric:
            fabric.backward(loss)
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        num_batches += 1

        # Progress every 20 batches
        if num_batches % 20 == 0:
            elapsed = time.time() - epoch_start
            spd = (num_batches * batch_size) / elapsed
            log(f'    batch {num_batches}/{total_batches_est} | '
                f'data {data_times[-1]*1000:.0f}ms | '
                f'fwd+bwd {batch_times[-1]*1000:.0f}ms | '
                f'{spd:.0f} samples/s')

        data_start = time.time()

    epoch_time = time.time() - epoch_start
    log(f'  [5/5] Epoch done: {num_batches} batches in {epoch_time:.1f}s')

    # Collect results
    result['epoch_time_s'] = epoch_time
    result['num_batches'] = num_batches
    result['avg_data_time_ms'] = np.mean(data_times[2:]) * 1000 if len(data_times) > 2 else 0
    result['avg_batch_time_ms'] = np.mean(batch_times[2:]) * 1000 if len(batch_times) > 2 else 0
    result['throughput_samples_per_sec'] = (num_batches * batch_size) / epoch_time if epoch_time > 0 else 0

    if torch.cuda.is_available():
        result['gpu_mem_peak_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return result


def print_results_table(results, rank=0):
    """Print formatted comparison table."""
    if rank != 0:
        return

    print('\n' + '=' * 90)
    print(f'{"Format":<12} {"Disk(MB)":<10} {"Epoch(s)":<10} {"DataLoad(ms)":<13} '
          f'{"Batch(ms)":<10} {"Throughput":<12} {"GPU Mem(MB)":<12}')
    print('-' * 90)

    for r in results:
        if 'error' in r:
            print(f'{r["format"]:<12} ERROR: {r["error"]}')
            continue
        print(f'{r["format"]:<12} '
              f'{r.get("disk_mb", 0):<10.1f} '
              f'{r.get("epoch_time_s", 0):<10.2f} '
              f'{r.get("avg_data_time_ms", 0):<13.2f} '
              f'{r.get("avg_batch_time_ms", 0):<10.2f} '
              f'{r.get("throughput_samples_per_sec", 0):<12.0f} '
              f'{r.get("gpu_mem_peak_mb", 0):<12.0f}')
    print('=' * 90)


def main():
    parser = argparse.ArgumentParser(description='Data format benchmark')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageFolder dataset (label dirs)')
    parser.add_argument('--benchmark_root', type=str, default='',
                        help='Path to bundled formats (default: data_root/../benchmark)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--formats', type=str, default='all',
                        help='Comma-separated formats or "all"')
    parser.add_argument('--model', type=str, default='ir18',
                        choices=['ir18', 'ir101'],
                        help='Model for benchmarking')
    parser.add_argument('--save_results', type=str, default='benchmark_results.json')
    args = parser.parse_args()

    if not args.benchmark_root:
        args.benchmark_root = os.path.join(os.path.dirname(args.data_root.rstrip('/')), 'benchmark')

    if args.formats == 'all':
        formats = ['imagefolder', 'lmdb', 'recordio', 'webdataset', 'hdf5', 'streaming']
    else:
        formats = [f.strip() for f in args.formats.split(',')]

    # Setup Fabric if available and in DDP context
    fabric = None
    if Fabric is not None and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        # Launched via `fabric run` — processes already created, do NOT call launch()
        fabric = Fabric()
    elif Fabric is not None and torch.cuda.is_available():
        fabric = Fabric(accelerator='cuda', devices=1)
        fabric.launch()

    rank = fabric.global_rank if fabric else 0

    if rank == 0:
        print(f'Benchmark config:')
        print(f'  Data root:      {args.data_root}')
        print(f'  Benchmark root: {args.benchmark_root}')
        print(f'  Batch size:     {args.batch_size}')
        print(f'  Num workers:    {args.num_workers}')
        print(f'  Model:          {args.model}')
        print(f'  Formats:        {formats}')
        print(f'  World size:     {fabric.world_size if fabric else 1}')
        print()

    transform = get_transform()

    # Load model
    model = get_model(args.model)

    # Run benchmarks
    all_results = []
    for fmt in formats:
        # if rank == 0:
        #     # Drop page cache before each format to measure cold IO
        #     print(f'\n  Dropping page cache...', flush=True)
        #     import subprocess
        #     subprocess.run(['bash', '-c', 'sync && echo 3 > /proc/sys/vm/drop_caches'],
        #                   check=False, timeout=10)
        #     print(f'\n--- Benchmarking: {fmt} ---', flush=True)

        # Synchronize before each test
        if fabric:
            fabric.barrier()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        result = benchmark_format(
            fmt=fmt,
            data_root=args.data_root,
            benchmark_root=args.benchmark_root,
            model=model,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fabric=fabric,
        )
        all_results.append(result)

        if rank == 0 and 'error' not in result:
            print(f'  Epoch time: {result["epoch_time_s"]:.2f}s, '
                  f'Data load: {result["avg_data_time_ms"]:.2f}ms/batch, '
                  f'Throughput: {result["throughput_samples_per_sec"]:.0f} samples/s')

    # Print final table
    print_results_table(all_results, rank)

    # Save results
    if rank == 0:
        save_path = os.path.join(args.benchmark_root, args.save_results)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nResults saved to: {save_path}')


if __name__ == '__main__':
    main()

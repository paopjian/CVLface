"""
Benchmark: DataLoader configurations under torch.compile + bf16.
Tests how persistent_workers, prefetch_factor, num_workers, and data format
affect throughput when GPU compute is fast (simulating train_opt.py).

Usage (single GPU):
    python benchmark_dataloader_opt.py --data_root /data1/dataset_0605/try2 \
        --benchmark_root /data1/dataset_0605/benchmark2

Usage (multi-GPU, closer to real training):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run \
        --strategy=ddp --devices=7 --precision="bf16-mixed" \
        benchmark_dataloader_opt.py --data_root /data1/dataset_0605/try2 \
        --benchmark_root /data1/dataset_0605/benchmark2
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
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

try:
    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy
except ImportError:
    Fabric = None

from dataset.benchmark_datasets import (
    ImageFolderFaceDataset, LMDBFaceDataset, RecordIOFaceDataset
)


def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_model():
    """Load iResNet-101 (same as train_opt.py)."""
    from models import get_model as _get_model
    from general_utils.config_utils import load_config
    yaml_path = 'models/iresnet/configs/v1_ir101.yaml'
    # Resolve relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_yaml_path = os.path.join(script_dir, yaml_path)
    model_cfg = load_config(abs_yaml_path)
    model_cfg.yaml_path = yaml_path
    model = _get_model(model_cfg, yaml_path)
    return model


def get_dataset(fmt, data_root, benchmark_root, transform):
    if fmt == 'imagefolder':
        return ImageFolderFaceDataset(data_root, transform=transform)
    elif fmt == 'lmdb':
        return LMDBFaceDataset(os.path.join(benchmark_root, 'lmdb'), transform=transform)
    elif fmt == 'recordio':
        return RecordIOFaceDataset(os.path.join(benchmark_root, 'recordio'), transform=transform)
    else:
        raise ValueError(f'Unknown format: {fmt}')


def make_dataloader(dataset, batch_size, num_workers, persistent_workers, prefetch_factor,
                    fabric=None):
    """Create DataLoader with specified config."""
    kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor

    if fabric and fabric.world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=fabric.world_size,
                                     rank=fabric.global_rank, shuffle=True)
        kwargs['sampler'] = sampler
    else:
        kwargs['shuffle'] = True

    return DataLoader(dataset, **kwargs)


def run_benchmark(config_name, dataset, batch_size, num_workers,
                  persistent_workers, prefetch_factor,
                  model, classifier, optimizer, criterion,
                  fabric, num_batches_limit=200, warmup_batches=10,
                  freeze_backbone=True):
    """Run a single benchmark configuration."""
    rank = fabric.global_rank if fabric else 0

    dataloader = make_dataloader(dataset, batch_size, num_workers,
                                 persistent_workers, prefetch_factor, fabric)
    if fabric:
        dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False
                                              if fabric.world_size > 1 else True)

    model.eval() if freeze_backbone else model.train()
    classifier.train()

    # Warmup
    data_times = []
    compute_times = []
    total_samples = 0

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    epoch_start = time.time()
    data_start = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches_limit + warmup_batches:
            break

        torch.cuda.synchronize()
        data_end = time.time()

        # Forward + backward
        compute_start = time.time()
        if freeze_backbone:
            with torch.no_grad():
                features = model(images)
            features = features.detach()
        else:
            features = model(images)
        logits = classifier(features)
        loss = criterion(logits, labels)
        if fabric:
            fabric.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        compute_end = time.time()

        # Skip warmup batches for timing
        if batch_idx >= warmup_batches:
            data_times.append(data_end - data_start)
            compute_times.append(compute_end - compute_start)
            total_samples += batch_size

        data_start = time.time()

    torch.cuda.synchronize()
    total_time = time.time() - epoch_start

    # Calculate metrics (skip warmup)
    actual_batches = len(data_times)
    if actual_batches == 0:
        return None

    world_size = fabric.world_size if fabric else 1
    result = {
        'config': config_name,
        'num_batches': actual_batches,
        'avg_data_ms': np.mean(data_times) * 1000,
        'p95_data_ms': np.percentile(data_times, 95) * 1000,
        'avg_compute_ms': np.mean(compute_times) * 1000,
        'throughput': total_samples / sum(data_times[i] + compute_times[i] for i in range(actual_batches)),
        'throughput_total': total_samples * world_size / sum(data_times[i] + compute_times[i] for i in range(actual_batches)),
        'gpu_util_pct': sum(compute_times) / (sum(data_times) + sum(compute_times)) * 100,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--benchmark_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_batches', type=int, default=200,
                        help='Number of batches to measure (after warmup)')
    parser.add_argument('--drop_cache', action='store_true',
                        help='Drop page cache before each test')
    args = parser.parse_args()

    if not args.benchmark_root:
        args.benchmark_root = os.path.join(os.path.dirname(args.data_root.rstrip('/')),
                                           'benchmark2')

    # Setup Fabric
    fabric = None
    if Fabric is not None and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        fabric = Fabric()
    elif Fabric is not None and torch.cuda.is_available():
        fabric = Fabric(accelerator='cuda', devices=1, precision='bf16-mixed')
        fabric.launch()

    rank = fabric.global_rank if fabric else 0
    device = fabric.device if fabric else torch.device('cuda')

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f'=== DataLoader Optimization Benchmark (torch.compile + bf16) ===')
    log(f'Data root: {args.data_root}')
    log(f'Benchmark root: {args.benchmark_root}')
    log(f'Batch size: {args.batch_size}')
    log(f'World size: {fabric.world_size if fabric else 1}')
    log(f'Num batches: {args.num_batches}')
    log('')

    transform = get_transform()

    # Load model + compile (same as train_opt.py)
    log('Loading iResNet-101 + torch.compile...')
    model = get_model()
    torch.backends.cudnn.benchmark = True
    # Freeze backbone (same as train_opt.py with models.freeze=True)
    for p in model.parameters():
        p.requires_grad = False
    model = torch.compile(model, dynamic=False)

    num_classes = len([d for d in os.listdir(args.data_root)
                       if os.path.isdir(os.path.join(args.data_root, d))])
    classifier = nn.Linear(512, num_classes)
    criterion = nn.CrossEntropyLoss()

    if fabric:
        # Only optimize classifier
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
        model = fabric.setup_module(model)
        classifier, optimizer = fabric.setup(classifier, optimizer)
    else:
        model = model.to(device)
        classifier = classifier.to(device)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Define test configurations
    test_configs = [
        # (name, format, num_workers, persistent_workers, prefetch_factor)
        ('ImgFolder_w8_base',       'imagefolder', 8,  False, None),
        ('ImgFolder_w8_persist',    'imagefolder', 8,  True,  None),
        ('ImgFolder_w8_pf3',        'imagefolder', 8,  True,  3),
        ('ImgFolder_w12_pf3',       'imagefolder', 12, True,  3),
        ('ImgFolder_w16_pf3',       'imagefolder', 16, True,  3),
        ('LMDB_w8_base',            'lmdb',        8,  False, None),
        ('LMDB_w8_persist',         'lmdb',        8,  True,  None),
        ('LMDB_w8_pf3',             'lmdb',        8,  True,  3),
        ('LMDB_w12_pf3',           'lmdb',        12, True,  3),
        ('LMDB_w16_pf3',           'lmdb',        16, True,  3),
        ('RecordIO_w8_base',        'recordio',    8,  False, None),
        ('RecordIO_w8_persist',     'recordio',    8,  True,  None),
        ('RecordIO_w8_pf3',         'recordio',    8,  True,  3),
        ('RecordIO_w12_pf3',        'recordio',    12, True,  3),
    ]

    # Run benchmarks
    results = []
    datasets_cache = {}

    for name, fmt, nw, pw, pf in test_configs:
        log(f'\n--- {name} ---')

        # Check format availability
        if fmt == 'lmdb' and not os.path.exists(os.path.join(args.benchmark_root, 'lmdb')):
            log(f'  SKIP: LMDB not found at {args.benchmark_root}/lmdb')
            continue
        if fmt == 'recordio' and not os.path.exists(os.path.join(args.benchmark_root, 'recordio')):
            log(f'  SKIP: RecordIO not found at {args.benchmark_root}/recordio')
            continue

        # Drop page cache if requested
        if args.drop_cache and rank == 0:
            os.system('sync && echo 3 > /proc/sys/vm/drop_caches')
            time.sleep(1)
        if fabric:
            fabric.barrier()

        # Get or create dataset
        if fmt not in datasets_cache:
            datasets_cache[fmt] = get_dataset(fmt, args.data_root, args.benchmark_root, transform)
        dataset = datasets_cache[fmt]

        try:
            result = run_benchmark(
                name, dataset, args.batch_size, nw, pw, pf,
                model, classifier, optimizer, criterion,
                fabric, num_batches_limit=args.num_batches
            )
            if result:
                results.append(result)
                log(f'  throughput: {result["throughput"]:.0f} img/s/gpu | '
                    f'total: {result["throughput_total"]:.0f} img/s | '
                    f'data: {result["avg_data_ms"]:.1f}ms (p95: {result["p95_data_ms"]:.1f}ms) | '
                    f'compute: {result["avg_compute_ms"]:.1f}ms | '
                    f'GPU util: {result["gpu_util_pct"]:.1f}%')
        except Exception as e:
            log(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    if rank == 0 and results:
        log(f'\n{"="*100}')
        log(f'{"Config":<25} {"Throughput/GPU":<15} {"Total":<12} {"Data(ms)":<10} '
            f'{"P95 Data":<10} {"Compute(ms)":<12} {"GPU Util%":<10}')
        log(f'{"-"*100}')
        for r in results:
            log(f'{r["config"]:<25} {r["throughput"]:<15.0f} {r["throughput_total"]:<12.0f} '
                f'{r["avg_data_ms"]:<10.1f} {r["p95_data_ms"]:<10.1f} '
                f'{r["avg_compute_ms"]:<12.1f} {r["gpu_util_pct"]:<10.1f}')
        log(f'{"="*100}')

        # Save results
        save_path = os.path.join(os.path.dirname(__file__), 'benchmark_dataloader_opt_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f'\nResults saved to {save_path}')


if __name__ == '__main__':
    main()

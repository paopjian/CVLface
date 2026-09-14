"""
Data format benchmark using MosaicML Composer.

Usage (single GPU):
    python benchmark_composer.py --data_root /data1/dataset_0605/try3

Usage (multi-GPU):
    composer -n 7 benchmark_composer.py --data_root /data1/dataset_0605/try3 \
        --benchmark_root /data1/dataset_0605/benchmark3

Requires: pip install mosaicml-streaming composer
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
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from composer import Trainer
from composer.models import ComposerModel
from composer.callbacks import SpeedMonitor
from composer.utils import dist

from streaming import StreamingDataset, StreamingDataLoader


def get_transform():
    """Standard face recognition transform: 112x112, normalize."""
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_backbone(model_name='ir18'):
    """Get a backbone model for benchmarking."""
    from models import get_model as _get_model
    from general_utils.config_utils import load_config

    if model_name == 'ir18':
        yaml_path = 'iresnet/configs/v1_ir18.yaml'
    else:
        yaml_path = 'iresnet/configs/v1_ir101.yaml'

    try:
        model_cfg = load_config(yaml_path)
        model = _get_model(model_cfg, yaml_path)
    except Exception:
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
        )
    return model


class FaceRecognitionModel(ComposerModel):
    """Composer wrapper for face recognition backbone + classifier."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, batch):
        images, _ = batch
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits

    def loss(self, outputs, batch):
        _, targets = batch
        return F.cross_entropy(outputs, targets)

    def get_metrics(self, is_train=False):
        return {}

    def eval_forward(self, batch, outputs=None):
        return self.forward(batch)

    def update_metric(self, batch, outputs, metric):
        pass


class FaceStreamingDataset(StreamingDataset):
    """StreamingDataset subclass for face recognition.

    Inherits from streaming.StreamingDataset (IterableDataset).
    Composer natively recognizes IterableDataset and skips DistributedSampler.
    StreamingDataset handles distributed sharding internally.
    """

    def __init__(self, local, transform=None, shuffle=True, batch_size=256):
        super().__init__(local=local, shuffle=shuffle, batch_size=batch_size)
        self.transform = transform

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img_bytes = sample['img']
        label = int(sample['label'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def benchmark_streaming(data_root, benchmark_root, backbone, transform,
                        batch_size, num_workers, num_classes=None):
    """Benchmark streaming format using Composer Trainer for 1 epoch."""
    result = {'format': 'streaming'}
    rank = dist.get_global_rank()

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # Dataset
    log(f'  [1/4] Creating StreamingDataset...')
    t0 = time.time()
    streaming_dir = os.path.join(benchmark_root, 'streaming')
    dataset = FaceStreamingDataset(
        local=streaming_dir,
        transform=transform,
        shuffle=True,
        batch_size=batch_size,
    )
    result['dataset_load_time'] = time.time() - t0
    result['num_samples'] = len(dataset)

    if num_classes is None:
        num_classes = len([d for d in os.listdir(data_root)
                          if os.path.isdir(os.path.join(data_root, d))])
    log(f'  [1/4] Dataset ready: {len(dataset)} samples, {num_classes} classes ({time.time()-t0:.1f}s)')

    # DataLoader — use StreamingDataLoader (native companion)
    log(f'  [2/4] Creating StreamingDataLoader (workers={num_workers})...')
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    log(f'  [2/4] DataLoader ready.')

    # Model
    log(f'  [3/4] Setting up Composer model...')
    model = FaceRecognitionModel(backbone, num_classes)
    log(f'  [3/4] Model ready.')

    # Trainer
    log(f'  [4/4] Training 1 epoch with Composer...')
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    epoch_start = time.time()

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        optimizers=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        max_duration='1ep',
        device='gpu',
        precision='amp_bf16',
        progress_bar=True,
        log_to_console=False,
        callbacks=[SpeedMonitor(window_size=100)],
    )
    trainer.fit()

    epoch_time = time.time() - epoch_start
    num_batches = len(dataset) // batch_size
    throughput = (num_batches * batch_size) / epoch_time if epoch_time > 0 else 0
    log(f'  Epoch done: {num_batches} batches in {epoch_time:.1f}s, '
        f'Throughput: {throughput:.0f} samples/s')

    result['epoch_time_s'] = epoch_time
    result['num_batches'] = num_batches
    result['throughput_samples_per_sec'] = throughput

    if torch.cuda.is_available():
        result['gpu_mem_peak_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return result


def main():
    parser = argparse.ArgumentParser(description='Data format benchmark (Composer)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--benchmark_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='ir18', choices=['ir18', 'ir101'])
    parser.add_argument('--save_results', type=str, default='benchmark_composer_results.json')
    args = parser.parse_args()

    if not args.benchmark_root:
        args.benchmark_root = os.path.join(os.path.dirname(args.data_root.rstrip('/')), 'benchmark3')

    # Initialize distributed
    dist.initialize_dist('gpu', timeout=300)

    rank = dist.get_global_rank()
    if rank == 0:
        print(f'Benchmark config (Composer + Streaming):')
        print(f'  Data root:      {args.data_root}')
        print(f'  Benchmark root: {args.benchmark_root}')
        print(f'  Batch size:     {args.batch_size}')
        print(f'  Num workers:    {args.num_workers}')
        print(f'  Model:          {args.model}')
        print(f'  World size:     {dist.get_world_size()}')
        print()

    transform = get_transform()
    backbone = get_backbone(args.model)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    result = benchmark_streaming(
        data_root=args.data_root,
        benchmark_root=args.benchmark_root,
        backbone=backbone,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Print result
    if rank == 0:
        print(f'\n{"="*80}')
        print(f'{"Format":<12} {"Epoch(s)":<10} {"Throughput":<14} {"GPU Mem(MB)":<12} {"DS Load(s)":<10}')
        print(f'{"-"*80}')
        print(f'{"streaming":<12} '
              f'{result.get("epoch_time_s", 0):<10.2f} '
              f'{result.get("throughput_samples_per_sec", 0):<14.0f} '
              f'{result.get("gpu_mem_peak_mb", 0):<12.0f} '
              f'{result.get("dataset_load_time", 0):<10.2f}')
        print(f'{"="*80}')

        save_path = os.path.join(args.benchmark_root, args.save_results)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nResults saved to: {save_path}')


if __name__ == '__main__':
    main()

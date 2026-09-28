"""
Fast bundling: all formats (LMDB, RecordIO, WebDataset, HDF5) from ImageFolder.
Optimization: reads raw JPEG bytes directly (no decode+re-encode).
Uses multiprocessing for file scanning.

Usage:
    python bundle_all_fast.py --source_dir /data1/dataset_0605/try3 --save_root /data1/dataset_0605/benchmark3
    python bundle_all_fast.py --source_dir /data1/dataset_0605/try3 --save_root /data1/dataset_0605/benchmark3 --formats lmdb,recordio
"""
import os
import re
import sys
import struct
import json
import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from io import BytesIO

import numpy as np
from tqdm import tqdm


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(s))]


def scan_one_dir(args):
    """Scan a single label directory for image files."""
    dirpath, label_name = args
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for f in os.listdir(dirpath):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(dirpath, f))
    return label_name, sorted(files)


def scan_dataset(source_dir):
    """Scan ImageFolder structure using multiprocessing. Returns (file_list, label_list, num_classes)."""
    print(f'Scanning {source_dir} ...')
    t0 = time.time()

    # Get all label directories
    entries = []
    for name in os.listdir(source_dir):
        d = os.path.join(source_dir, name)
        if os.path.isdir(d):
            entries.append((d, name))
    entries.sort(key=lambda x: natural_sort_key(x[1]))

    # Build label map
    label_names = [e[1] for e in entries]
    label_map = {name: idx for idx, name in enumerate(label_names)}
    num_classes = len(label_names)

    # Parallel scan
    with Pool(min(cpu_count(), 64)) as pool:
        results = list(tqdm(
            pool.imap_unordered(scan_one_dir, entries, chunksize=256),
            total=len(entries), desc='Scanning dirs', unit='dir'
        ))

    # Flatten and sort
    file_list = []
    label_list = []
    # Sort results by label_name to ensure consistent ordering
    results.sort(key=lambda x: natural_sort_key(x[0]))
    for label_name, files in results:
        label = label_map[label_name]
        for f in files:
            file_list.append(f)
            label_list.append(label)

    elapsed = time.time() - t0
    print(f'Found {len(file_list):,} images, {num_classes:,} classes in {elapsed:.1f}s')
    return file_list, label_list, num_classes, label_map


def bundle_lmdb(file_list, label_list, num_classes, label_map, save_dir):
    """Bundle into LMDB (raw bytes, no re-encode)."""
    import lmdb
    os.makedirs(save_dir, exist_ok=True)
    lmdb_path = os.path.join(save_dir, 'train.lmdb')
    print(f'\n[LMDB] Writing to {lmdb_path}')

    map_size = len(file_list) * 8000  # ~8KB avg per image
    map_size = max(map_size, 1 << 30)  # at least 1GB
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)

    t0 = time.time()
    for i, (path, label) in enumerate(tqdm(zip(file_list, label_list), total=len(file_list), desc='LMDB')):
        with open(path, 'rb') as f:
            img_bytes = f.read()
        key = f'{i:08d}'.encode('ascii')
        value = struct.pack('<i', label) + img_bytes
        txn.put(key, value)

        if (i + 1) % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b'__len__', str(len(file_list)).encode())
    txn.put(b'__num_classes__', str(num_classes).encode())
    txn.put(b'__label_map__', json.dumps(label_map).encode())
    txn.commit()
    env.close()

    elapsed = time.time() - t0
    size_gb = sum(os.path.getsize(os.path.join(save_dir, f)) for f in os.listdir(save_dir)) / 1e9
    print(f'[LMDB] Done: {elapsed:.1f}s, {size_gb:.2f} GB')


def bundle_recordio(file_list, label_list, num_classes, label_map, save_dir):
    """Bundle into RecordIO (pure Python, no mxnet)."""
    os.makedirs(save_dir, exist_ok=True)
    rec_path = os.path.join(save_dir, 'train.rec')
    idx_path = os.path.join(save_dir, 'train.idx')
    print(f'\n[RecordIO] Writing to {rec_path}')

    kMagic = 0xced7230a

    t0 = time.time()
    idx_file = open(idx_path, 'w')
    rec_file = open(rec_path, 'wb')

    def write_record(idx, data):
        offset = rec_file.tell()
        idx_file.write(f'{idx}\t{offset}\n')
        length = len(data)
        lrecord = (0 << 29) | length
        rec_file.write(struct.pack('<II', kMagic, lrecord))
        rec_file.write(data)
        pad = (4 - (length % 4)) % 4
        if pad:
            rec_file.write(b'\x00' * pad)

    # Header record at index 0
    n = len(file_list) + 1
    header_data = struct.pack('<IfQQ', 2, 0.0, 0, 0)
    header_data += struct.pack('<ff', float(n), float(n))
    write_record(0, header_data)

    for i, (path, label) in enumerate(tqdm(zip(file_list, label_list), total=len(file_list), desc='RecordIO')):
        with open(path, 'rb') as f:
            img_bytes = f.read()
        # IRHeader: flag=0, label, id=0, id2=0
        data = struct.pack('<IfQQ', 0, float(label), 0, 0) + img_bytes
        write_record(i + 1, data)

    idx_file.close()
    rec_file.close()

    # Save metadata
    meta = {'num_classes': num_classes, 'num_samples': len(file_list)}
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    size_gb = os.path.getsize(rec_path) / 1e9
    print(f'[RecordIO] Done: {elapsed:.1f}s, {size_gb:.2f} GB')


def bundle_webdataset(file_list, label_list, num_classes, label_map, save_dir, max_shard_size=100_000_000):
    """Bundle into WebDataset tar shards (raw bytes)."""
    import webdataset as wds
    os.makedirs(save_dir, exist_ok=True)
    shard_pattern = os.path.join(save_dir, 'shard-%06d.tar')
    print(f'\n[WebDataset] Writing to {save_dir}')

    t0 = time.time()
    sink = wds.ShardWriter(shard_pattern, maxsize=max_shard_size)

    for i, (path, label) in enumerate(tqdm(zip(file_list, label_list), total=len(file_list), desc='WebDataset')):
        with open(path, 'rb') as f:
            img_bytes = f.read()
        sample = {
            '__key__': f'{i:08d}',
            'jpg': img_bytes,
            'cls': str(label).encode(),
        }
        sink.write(sample)

    sink.close()

    meta = {'num_classes': num_classes, 'num_samples': len(file_list), 'label_map': label_map}
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    size_gb = sum(os.path.getsize(os.path.join(save_dir, f))
                  for f in os.listdir(save_dir) if f.endswith('.tar')) / 1e9
    print(f'[WebDataset] Done: {elapsed:.1f}s, {size_gb:.2f} GB')


def bundle_streaming(file_list, label_list, num_classes, label_map, save_dir):
    """Bundle into MosaicML Streaming (MDS) format."""
    from streaming import MDSWriter
    os.makedirs(save_dir, exist_ok=True)
    print(f'\n[Streaming] Writing to {save_dir}')

    columns = {
        'img': 'bytes',
        'label': 'int',
    }

    t0 = time.time()
    with MDSWriter(out=save_dir, columns=columns, compression='zstd',
                   size_limit=1 << 27) as writer:  # 128MB shards
        for i, (path, label) in enumerate(tqdm(zip(file_list, label_list),
                                               total=len(file_list), desc='Streaming')):
            with open(path, 'rb') as f:
                img_bytes = f.read()
            writer.write({'img': img_bytes, 'label': label})

    # Save extra metadata
    meta = {'num_classes': num_classes, 'num_samples': len(file_list)}
    with open(os.path.join(save_dir, 'extra_meta.json'), 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    size_gb = sum(os.path.getsize(os.path.join(save_dir, f))
                  for f in os.listdir(save_dir) if not f.endswith('.json')) / 1e9
    print(f'[Streaming] Done: {elapsed:.1f}s, {size_gb:.2f} GB')


def bundle_hdf5(file_list, label_list, num_classes, label_map, save_dir):
    """Bundle into HDF5 (raw JPEG bytes as variable-length arrays)."""
    import h5py
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, 'train.h5')
    print(f'\n[HDF5] Writing to {h5_path}')

    dt = h5py.vlen_dtype(np.dtype('uint8'))
    t0 = time.time()

    with h5py.File(h5_path, 'w') as f:
        images_ds = f.create_dataset('images', shape=(len(file_list),), dtype=dt)
        labels_ds = f.create_dataset('labels', shape=(len(file_list),), dtype='int32')
        f.attrs['num_classes'] = num_classes
        f.attrs['num_samples'] = len(file_list)

        for i, (path, label) in enumerate(tqdm(zip(file_list, label_list), total=len(file_list), desc='HDF5')):
            with open(path, 'rb') as fp:
                img_bytes = fp.read()
            images_ds[i] = np.frombuffer(img_bytes, dtype=np.uint8)
            labels_ds[i] = label

    elapsed = time.time() - t0
    size_gb = os.path.getsize(h5_path) / 1e9
    print(f'[HDF5] Done: {elapsed:.1f}s, {size_gb:.2f} GB')


def main():
    parser = argparse.ArgumentParser(description='Fast bundle all formats')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='ImageFolder source (label dirs with images)')
    parser.add_argument('--save_root', type=str, required=True,
                        help='Root dir for bundled outputs')
    parser.add_argument('--formats', type=str, default='lmdb,recordio,webdataset,hdf5,streaming',
                        help='Comma-separated formats to bundle')
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(',')]
    print(f'Formats to bundle: {formats}')
    print(f'Source: {args.source_dir}')
    print(f'Save root: {args.save_root}')

    # Scan dataset once
    file_list, label_list, num_classes, label_map = scan_dataset(args.source_dir)

    # Bundle each format
    for fmt in formats:
        save_dir = os.path.join(args.save_root, fmt)
        if fmt == 'lmdb':
            bundle_lmdb(file_list, label_list, num_classes, label_map, save_dir)
        elif fmt == 'recordio':
            bundle_recordio(file_list, label_list, num_classes, label_map, save_dir)
        elif fmt == 'webdataset':
            bundle_webdataset(file_list, label_list, num_classes, label_map, save_dir)
        elif fmt == 'streaming':
            bundle_streaming(file_list, label_list, num_classes, label_map, save_dir)
        elif fmt == 'hdf5':
            bundle_hdf5(file_list, label_list, num_classes, label_map, save_dir)
        else:
            print(f'Unknown format: {fmt}, skipping')

    # Summary
    print(f'\n{"="*60}')
    print('Disk usage:')
    print(f'  ImageFolder: {sum(os.path.getsize(f) for f in file_list) / 1e9:.2f} GB')
    for fmt in formats:
        d = os.path.join(args.save_root, fmt)
        if os.path.exists(d):
            size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e9
            print(f'  {fmt:12s}: {size:.2f} GB')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

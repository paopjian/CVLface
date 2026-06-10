"""
打包 /data1/dataset_0605/train 为 RecordIO 格式
纯顺序追加写入, 37M 图片约 7-8 分钟

用法:
    conda run --no-capture-output -n cvlface python scripts/benchmark/bundle_rec_large.py
"""
import os
import re
import struct
import json
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


SOURCE_DIR = '/data1/dataset_0605/train'
SAVE_DIR = '/data1/dataset_0605/train_rec'

kMagic = 0xced7230a


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(s))]


def scan_one_dir(args):
    dirpath, label_name = args
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for f in os.listdir(dirpath):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(dirpath, f))
    return label_name, sorted(files)


def scan_dataset(source_dir):
    print(f'Scanning {source_dir} ...')
    t0 = time.time()

    entries = []
    for name in os.listdir(source_dir):
        d = os.path.join(source_dir, name)
        if os.path.isdir(d):
            entries.append((d, name))
    entries.sort(key=lambda x: natural_sort_key(x[1]))

    label_names = [e[1] for e in entries]
    label_map = {name: idx for idx, name in enumerate(label_names)}
    num_classes = len(label_names)

    n_workers = min(cpu_count(), 64)
    print(f'Using {n_workers} workers to scan {num_classes} directories...')

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(scan_one_dir, entries, chunksize=512),
            total=len(entries), desc='Scanning dirs', unit='dir'
        ))

    results.sort(key=lambda x: natural_sort_key(x[0]))

    file_list = []
    label_list = []
    for label_name, files in results:
        label = label_map[label_name]
        for f in files:
            file_list.append(f)
            label_list.append(label)

    elapsed = time.time() - t0
    print(f'Found {len(file_list):,} images, {num_classes:,} classes in {elapsed:.1f}s')
    return file_list, label_list, num_classes, label_map


def bundle_recordio(file_list, label_list, num_classes, label_map, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    rec_path = os.path.join(save_dir, 'train.rec')
    idx_path = os.path.join(save_dir, 'train.idx')
    print(f'\n[RecordIO] Writing to {rec_path}')
    print(f'  {len(file_list):,} images, {num_classes:,} classes')

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

    for i, (path, label) in enumerate(tqdm(zip(file_list, label_list),
                                           total=len(file_list), desc='RecordIO')):
        with open(path, 'rb') as f:
            img_bytes = f.read()
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
    print(f'[RecordIO] Done: {elapsed:.1f}s ({elapsed/60:.1f} min), size: {size_gb:.1f} GB')
    print(f'  速度: {len(file_list)/elapsed:.0f} images/s')


if __name__ == '__main__':
    file_list, label_list, num_classes, label_map = scan_dataset(SOURCE_DIR)
    bundle_recordio(file_list, label_list, num_classes, label_map, SAVE_DIR)
    print('\n完成! RecordIO 路径:', SAVE_DIR)

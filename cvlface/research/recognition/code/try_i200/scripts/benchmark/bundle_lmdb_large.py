"""
打包 /data1/dataset_0605/train 为 LMDB 格式 (针对大数据集优化)

用法:
    conda run --no-capture-output -n cvlface python scripts/benchmark/bundle_lmdb_large.py

特点:
- 多进程扫描目录 (791K 类)
- 直接读原始 JPEG bytes (不 decode/re-encode)
- 每 50000 条 commit 一次, 避免内存积累
- map_size 设为 300GB (足够 37M 图片)
"""
import os
import re
import struct
import json
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import lmdb


SOURCE_DIR = '/data1/dataset_0605/train'
SAVE_DIR = '/data1/dataset_0605/train_lmdb'
MAP_SIZE = 300 * 1024 * 1024 * 1024  # 300GB
COMMIT_EVERY = 100000


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(s))]


def scan_one_dir(args):
    """扫描单个 label 目录"""
    dirpath, label_name = args
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for f in os.listdir(dirpath):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(dirpath, f))
    return label_name, sorted(files)


def scan_dataset(source_dir):
    """多进程扫描 ImageFolder 结构"""
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

    # 按 label_name 排序保证顺序一致
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


def bundle_lmdb(file_list, label_list, num_classes, label_map, save_dir):
    """写入 LMDB (只最后 commit 一次, writemap 模式)"""
    os.makedirs(save_dir, exist_ok=True)
    lmdb_path = os.path.join(save_dir, 'train.lmdb')
    print(f'\n[LMDB] Writing to {lmdb_path}')
    print(f'  map_size: {MAP_SIZE / 1e9:.0f} GB')
    print(f'  mode: writemap + 只最后 commit 一次')

    env = lmdb.open(lmdb_path, map_size=MAP_SIZE,
                    writemap=True, map_async=True, metasync=False)
    txn = env.begin(write=True)

    t0 = time.time()
    for i, (path, label) in enumerate(tqdm(zip(file_list, label_list),
                                           total=len(file_list), desc='LMDB')):
        with open(path, 'rb') as f:
            img_bytes = f.read()
        key = f'{i:08d}'.encode('ascii')
        value = struct.pack('<i', label) + img_bytes
        txn.put(key, value)

    # 写入元数据
    txn.put(b'__len__', str(len(file_list)).encode())
    txn.put(b'__num_classes__', str(num_classes).encode())
    txn.put(b'__label_map__', json.dumps(label_map).encode())

    # 只 commit 一次
    print('Committing (final sync)...')
    txn.commit()
    env.close()

    elapsed = time.time() - t0
    # 计算最终大小
    lmdb_data = os.path.join(lmdb_path, 'data.mdb')
    size_gb = os.path.getsize(lmdb_data) / 1e9 if os.path.exists(lmdb_data) else 0
    print(f'[LMDB] Done: {elapsed:.1f}s ({elapsed/60:.1f} min), size: {size_gb:.1f} GB')
    print(f'  速度: {len(file_list)/elapsed:.0f} images/s')


if __name__ == '__main__':
    file_list, label_list, num_classes, label_map = scan_dataset(SOURCE_DIR)
    bundle_lmdb(file_list, label_list, num_classes, label_map, SAVE_DIR)
    print('\n完成! LMDB 路径:', os.path.join(SAVE_DIR, 'train.lmdb'))

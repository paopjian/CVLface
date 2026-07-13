"""
Bundle images into RecordIO format (no mxnet dependency).
Compatible with mxnet RecordIO: no header record, index starts from 0, uniform JPEG encoding.
8 reader threads + sequential write, ~13K-15K img/s on NVMe.

Usage:
    python bundle_images_into_rec_v2.py --source_dir /path/to/images
    python bundle_images_into_rec_v2.py --source_dir /path/to/images --save_dir /path/to/output
    python bundle_images_into_rec_v2.py --source_dir /path/to/images --remove_images

Images should be stored in a directory structure where each subfolder is named
after the label and contains images for that label:
    source_dir/label1/image1.jpg
    source_dir/label2/image2.png
"""
import os
import re
import io
import struct
import json
import time
import argparse
from multiprocessing import Pool, cpu_count
from threading import Thread
from queue import Queue
from tqdm import tqdm
from PIL import Image


kMagic = 0xced7230a
JPEG_MAGIC = b'\xff\xd8\xff'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
NUM_READERS = 8


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(s))]


def scan_one_dir(args):
    dirpath, label_name = args
    files = []
    for f in os.listdir(dirpath):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(dirpath, f))
    return label_name, sorted(files)


def scan_dataset(source_dir):
    print(f'Scanning {source_dir} ...')
    t0 = time.time()

    entries = []
    for name in os.listdir(source_dir):
        d = os.path.join(source_dir, name)
        if os.path.isdir(d) and name != 'examples':
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
    return file_list, label_list, num_classes, label_names


def ensure_jpeg(img_bytes):
    """非 JPEG 文件重编码为 JPEG q=100"""
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=100)
    return buf.getvalue()


def reader_worker(file_list, label_list, start, end, out_queue):
    """读线程: 顺序读取 [start, end) 的文件, 放入队列"""
    try:
        for i in range(start, end):
            try:
                with open(file_list[i], 'rb') as f:
                    img_bytes = f.read()
                if img_bytes[:3] != JPEG_MAGIC:
                    img_bytes = ensure_jpeg(img_bytes)
                out_queue.put((i, label_list[i], img_bytes))
            except Exception:
                # 跳过无法读取/解码的图片
                pass
    finally:
        out_queue.put(None)  # sentinel 必须发出，否则主线程死锁


def bundle_recordio(file_list, label_list, num_classes, save_dir, label_names=None):
    os.makedirs(save_dir, exist_ok=True)
    rec_path = os.path.join(save_dir, 'train.rec')
    idx_path = os.path.join(save_dir, 'train.idx')
    tsv_path = os.path.join(save_dir, 'train.tsv')
    orig_path = os.path.join(save_dir, 'original_structure.txt')
    N = len(file_list)
    print(f'\n[RecordIO] Writing to {rec_path}')
    print(f'  {N:,} images, {num_classes:,} classes, {NUM_READERS} reader threads')

    # 删除已有文件
    for p in [rec_path, idx_path, tsv_path, orig_path]:
        if os.path.isfile(p):
            os.remove(p)

    t0 = time.time()
    rec_file = open(rec_path, 'wb', buffering=8*1024*1024)
    idx_file = open(idx_path, 'w', buffering=1*1024*1024)
    tsv_file = open(tsv_path, 'w', buffering=1*1024*1024)
    orig_file = open(orig_path, 'w', buffering=1*1024*1024)

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

    # 分段预读: 每个读线程负责一段连续范围, 按段顺序消费保证写入有序
    chunk_size = (N + NUM_READERS - 1) // NUM_READERS
    queues = []
    threads = []
    for r in range(NUM_READERS):
        start = r * chunk_size
        end = min(start + chunk_size, N)
        q = Queue(maxsize=256)
        queues.append(q)
        t = Thread(target=reader_worker, args=(file_list, label_list, start, end, q))
        threads.append(t)

    for t in threads:
        t.start()

    # 按段顺序消费
    # 尝试输出到真实终端，失败则静默禁用
    try:
        tty = open('/dev/tty', 'w')
        pbar = tqdm(total=N, desc='Writing RecordIO', ncols=80, file=tty)
    except (OSError, IOError):
        # 没有终端（非交互环境），禁用进度条
        pbar = tqdm(total=N, desc='Writing RecordIO', disable=True)

    for q in queues:
        while True:
            item = q.get()
            if item is None:
                break
            i, label, img_bytes = item
            # record: flag=0, label=float, id=index, id2=0
            data = struct.pack('<IfQQ', 0, float(label), i, 0) + img_bytes
            write_record(i, data)
            # tsv: image_index \t label/filename \t label
            filename = os.path.basename(file_list[i])
            tsv_file.write(f'{i}\t{label}/{filename}\t{label}\n')
            # original_structure: image_index \t original_path
            orig_file.write(f'{i}\t{file_list[i]}\n')
            pbar.update(1)
    pbar.close()

    for t in threads:
        t.join()
    idx_file.close()
    rec_file.close()
    tsv_file.close()
    orig_file.close()

    # Save metadata
    meta = {'num_classes': num_classes, 'num_samples': N}
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    size_gb = os.path.getsize(rec_path) / 1e9
    print(f'[RecordIO] Done: {elapsed:.1f}s ({elapsed/60:.1f} min), size: {size_gb:.1f} GB')
    print(f'  速度: {N/elapsed:.0f} images/s')
    print(f'[TSV] {tsv_path} ({N:,} entries)')
    print(f'[Original Structure] {orig_path} ({N:,} entries)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bundle images into RecordIO format (no mxnet dependency). '
                    'Images should be stored in a directory structure where each '
                    'subfolder is named after the label and contains images for that label, '
                    'e.g., label1/image1.png, label2/image2.png.')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Directory containing labeled image folders.')
    parser.add_argument('--save_dir', default='', type=str,
                        help='Output directory. Defaults to source_dir.')
    parser.add_argument('--remove_images', action='store_true',
                        help='Remove source image directories after bundling.')

    args = parser.parse_args()
    source_dir = args.source_dir.rstrip('/')
    if not args.save_dir:
        save_dir = source_dir
    else:
        save_dir = args.save_dir

    file_list, label_list, num_classes, label_names = scan_dataset(source_dir)
    bundle_recordio(file_list, label_list, num_classes, save_dir, label_names)

    # remove source image dirs
    if args.remove_images:
        import shutil
        for d in os.listdir(source_dir):
            full = os.path.join(source_dir, d)
            if os.path.isdir(full) and d != 'examples':
                shutil.rmtree(full)
        print('Source image directories removed.')

    print('\n完成! RecordIO 路径:', save_dir)

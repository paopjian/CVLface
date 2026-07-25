"""Bundle images from label-folder structure into LMDB format."""
import os
import sys
import re
import struct
import json
import argparse
import time
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from tqdm import tqdm


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)


def get_all_images(root_dir):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(dirpath, f))
    return natural_sort(files)


def main():
    parser = argparse.ArgumentParser(description='Bundle images into LMDB')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--quality', type=int, default=100)
    parser.add_argument('--map_size', type=int, default=1 << 40)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    lmdb_path = os.path.join(args.save_dir, 'train.lmdb')

    all_images = get_all_images(args.source_dir)
    print(f'Found {len(all_images)} images')

    # Build label mapping from directory names
    labels_set = set()
    for p in all_images:
        rel = os.path.relpath(p, args.source_dir)
        labels_set.add(rel.split(os.sep)[0])
    label_names = natural_sort(list(labels_set))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    num_classes = len(label_names)
    print(f'Classes: {num_classes}')

    start = time.time()
    env = lmdb.open(lmdb_path, map_size=args.map_size)
    txn = env.begin(write=True)

    for i, path in enumerate(tqdm(all_images, desc='LMDB')):
        rel = os.path.relpath(path, args.source_dir)
        label_name = rel.split(os.sep)[0]
        label = label_map[label_name]

        img = Image.open(path).convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=args.quality)
        img_bytes = buf.getvalue()

        key = f'{i:08d}'.encode('ascii')
        value = struct.pack('<i', label) + img_bytes
        txn.put(key, value)

        if (i + 1) % 5000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    # Metadata
    txn.put(b'__len__', str(len(all_images)).encode())
    txn.put(b'__num_classes__', str(num_classes).encode())
    txn.put(b'__label_map__', json.dumps(label_map).encode())
    txn.commit()
    env.close()

    elapsed = time.time() - start
    print(f'Done. Time: {elapsed:.1f}s, Path: {lmdb_path}')


if __name__ == '__main__':
    main()

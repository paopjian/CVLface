"""Bundle images into WebDataset tar shards."""
import os
import re
import argparse
import time
import json
from io import BytesIO

import webdataset as wds
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
    parser = argparse.ArgumentParser(description='Bundle images into WebDataset')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--quality', type=int, default=100)
    parser.add_argument('--max_shard_size', type=int, default=100_000_000,
                        help='Max shard size in bytes (default 100MB)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    shard_pattern = os.path.join(args.save_dir, 'shard-%06d.tar')

    all_images = get_all_images(args.source_dir)
    print(f'Found {len(all_images)} images')

    labels_set = set()
    for p in all_images:
        rel = os.path.relpath(p, args.source_dir)
        labels_set.add(rel.split(os.sep)[0])
    label_names = natural_sort(list(labels_set))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    num_classes = len(label_names)
    print(f'Classes: {num_classes}')

    start = time.time()
    sink = wds.ShardWriter(shard_pattern, maxsize=args.max_shard_size)

    for i, path in enumerate(tqdm(all_images, desc='WebDataset')):
        rel = os.path.relpath(path, args.source_dir)
        label_name = rel.split(os.sep)[0]
        label = label_map[label_name]

        img = Image.open(path).convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=args.quality)

        sample = {
            '__key__': f'{i:08d}',
            'jpg': buf.getvalue(),
            'cls': str(label).encode(),
        }
        sink.write(sample)

    sink.close()

    # Save metadata
    meta = {'num_classes': num_classes, 'num_samples': len(all_images),
            'label_map': label_map}
    with open(os.path.join(args.save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    elapsed = time.time() - start
    print(f'Done. Time: {elapsed:.1f}s, Path: {args.save_dir}')


if __name__ == '__main__':
    main()

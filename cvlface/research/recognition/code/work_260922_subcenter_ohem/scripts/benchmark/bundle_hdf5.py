"""Bundle images into HDF5 format."""
import os
import re
import argparse
import time
from io import BytesIO

import h5py
import numpy as np
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
    parser = argparse.ArgumentParser(description='Bundle images into HDF5')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--quality', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    h5_path = os.path.join(args.save_dir, 'train.h5')

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

    # Store as variable-length byte strings (JPEG encoded)
    dt = h5py.vlen_dtype(np.dtype('uint8'))

    with h5py.File(h5_path, 'w') as f:
        images_ds = f.create_dataset('images', shape=(len(all_images),), dtype=dt)
        labels_ds = f.create_dataset('labels', shape=(len(all_images),), dtype='int32')
        f.attrs['num_classes'] = num_classes
        f.attrs['num_samples'] = len(all_images)

        for i, path in enumerate(tqdm(all_images, desc='HDF5')):
            rel = os.path.relpath(path, args.source_dir)
            label_name = rel.split(os.sep)[0]
            label = label_map[label_name]

            img = Image.open(path).convert('RGB')
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=args.quality)
            img_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)

            images_ds[i] = img_bytes
            labels_ds[i] = label

    elapsed = time.time() - start
    print(f'Done. Time: {elapsed:.1f}s, Path: {h5_path}')


if __name__ == '__main__':
    main()

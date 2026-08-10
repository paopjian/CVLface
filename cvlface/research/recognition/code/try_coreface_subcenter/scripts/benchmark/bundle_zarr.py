"""Bundle images into Zarr format (raw pixels, LZ4 compression, batch writes)."""
import os
import re
import argparse
import time

import numpy as np
import zarr
from zarr.codecs import BytesCodec, BloscCodec
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
    parser = argparse.ArgumentParser(description='Bundle images into Zarr')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--quality', type=int, default=100)
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='Zarr chunk size (match training batch_size for best read perf)')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Number of images to buffer before flushing to disk')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    zarr_path = os.path.join(args.save_dir, 'train.zarr')

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

    # Peek at first image to get shape
    first_img = np.array(Image.open(all_images[0]).convert('RGB'))
    h, w, c = first_img.shape
    print(f'Image shape: {h}x{w}x{c}')

    start = time.time()

    # Use LZ4 compression (fast) with chunk_size matching training batch_size
    chunk_size = min(args.chunk_size, len(all_images))

    store = zarr.open_group(zarr_path, mode='w')
    images = store.create_array('images', shape=(len(all_images), h, w, c),
                                chunks=(chunk_size, h, w, c), dtype='uint8',
                                serializer=BytesCodec(),
                                compressors=BloscCodec(cname='lz4', clevel=1))
    labels_arr = store.create_array('labels', shape=(len(all_images),),
                                    chunks=(args.chunk_size,), dtype='int32')
    store.attrs['num_classes'] = num_classes
    store.attrs['num_samples'] = len(all_images)

    # Batch write: accumulate images in memory then flush as a block
    batch_imgs = np.empty((args.batch_size, h, w, c), dtype=np.uint8)
    batch_labels = np.empty(args.batch_size, dtype=np.int32)
    batch_idx = 0
    global_idx = 0

    for i, path in enumerate(tqdm(all_images, desc='Zarr (LZ4)')):
        rel = os.path.relpath(path, args.source_dir)
        label_name = rel.split(os.sep)[0]
        label = label_map[label_name]

        img = np.array(Image.open(path).convert('RGB'))
        if img.shape != (h, w, c):
            img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

        batch_imgs[batch_idx] = img
        batch_labels[batch_idx] = label
        batch_idx += 1

        # Flush batch
        if batch_idx == args.batch_size:
            images[global_idx:global_idx + batch_idx] = batch_imgs[:batch_idx]
            labels_arr[global_idx:global_idx + batch_idx] = batch_labels[:batch_idx]
            global_idx += batch_idx
            batch_idx = 0

    # Flush remaining
    if batch_idx > 0:
        images[global_idx:global_idx + batch_idx] = batch_imgs[:batch_idx]
        labels_arr[global_idx:global_idx + batch_idx] = batch_labels[:batch_idx]

    elapsed = time.time() - start
    print(f'Done. Time: {elapsed:.1f}s, Path: {zarr_path}')


if __name__ == '__main__':
    main()

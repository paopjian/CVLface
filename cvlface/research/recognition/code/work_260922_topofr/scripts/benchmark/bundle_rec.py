"""Bundle images into pure-Python RecordIO format (no mxnet dependency)."""
import os
import sys
import re
import struct
import argparse
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

kMagic = 0xced7230a


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


class RecordIOWriter:
    """Write MXNet-compatible RecordIO .rec/.idx files using pure Python."""

    def __init__(self, idx_path, rec_path):
        self.idx_file = open(idx_path, 'w')
        self.rec_file = open(rec_path, 'wb')

    def write_record(self, idx, data):
        """Write a single record. data is bytes."""
        offset = self.rec_file.tell()
        self.idx_file.write(f'{idx}\t{offset}\n')

        length = len(data)
        # cflag=0 (single record), length
        lrecord = (0 << 29) | length
        self.rec_file.write(struct.pack('<II', kMagic, lrecord))
        self.rec_file.write(data)

        # Pad to 4-byte boundary
        pad = (4 - (length % 4)) % 4
        if pad:
            self.rec_file.write(b'\x00' * pad)

    def close(self):
        self.idx_file.close()
        self.rec_file.close()


def pack_header_and_image(label, img_bytes):
    """Pack IRHeader (flag=0, single label) + image bytes."""
    # IRHeader: flag(uint32) + label(float32) + id(uint64) + id2(uint64) = 24 bytes
    header = struct.pack('<IfQQ', 0, float(label), 0, 0)
    return header + img_bytes


def main():
    parser = argparse.ArgumentParser(description='Bundle images into RecordIO')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--quality', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    rec_path = os.path.join(args.save_dir, 'train.rec')
    idx_path = os.path.join(args.save_dir, 'train.idx')

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
    writer = RecordIOWriter(idx_path, rec_path)

    # Write header record at index 0 (contains num_samples info)
    # flag > 0 means label is an array: [num_samples, num_samples]
    header_data = struct.pack('<IfQQ', 2, 0.0, 0, 0)
    # Two float32 labels: [num_samples, num_samples] (mimics mxnet convention)
    n = len(all_images) + 1  # indices start at 1
    header_data += struct.pack('<ff', float(n), float(n))
    writer.write_record(0, header_data)

    for i, path in enumerate(tqdm(all_images, desc='RecordIO')):
        rel = os.path.relpath(path, args.source_dir)
        label_name = rel.split(os.sep)[0]
        label = label_map[label_name]

        img = Image.open(path).convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=args.quality)
        img_bytes = buf.getvalue()

        data = pack_header_and_image(label, img_bytes)
        writer.write_record(i + 1, data)

    writer.close()

    elapsed = time.time() - start
    print(f'Done. Time: {elapsed:.1f}s, Path: {rec_path}')


if __name__ == '__main__':
    main()

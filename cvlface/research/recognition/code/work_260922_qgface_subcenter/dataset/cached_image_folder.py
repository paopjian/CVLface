import os
import time
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from general_utils import dist_utils


class CachedImageFolderDataset(Dataset):
    """ImageFolder dataset with parquet-cached index for fast startup.

    On first run (rank 0): scans the directory tree, builds (relative_path, label) index,
    saves as _cached_index.parquet with zstd compression.
    On subsequent runs: loads the parquet file directly (~5-15s for 37M entries).
    """

    CACHE_FILENAME = '_cached_index.parquet'
    BUILDING_SENTINEL = '_cached_index.building'

    def __init__(self, root_dir, transform=None, local_rank=0):
        assert HAS_PYARROW, "pyarrow is required for CachedImageFolderDataset. Install with: pip install pyarrow"

        self.root_dir = root_dir
        self.transform = transform
        self.local_rank = local_rank
        self.color_space = 'RGB'

        cache_path = os.path.join(root_dir, self.CACHE_FILENAME)
        sentinel_path = os.path.join(root_dir, self.BUILDING_SENTINEL)

        # Rank 0 builds cache if needed
        if local_rank == 0:
            if not os.path.exists(cache_path) or os.path.exists(sentinel_path):
                self._build_cache(cache_path, sentinel_path)

        # Other ranks wait for rank 0
        dist_utils.barrier()

        # Load cache
        if local_rank == 0:
            print(f"[CachedImageFolder] Loading index from {cache_path}")
        t0 = time.time()
        table = pq.read_table(cache_path)
        self.paths = table.column('path').to_pylist()
        self.labels = table.column('label').to_numpy()
        elapsed = time.time() - t0
        if local_rank == 0:
            print(f"[CachedImageFolder] Loaded {len(self.paths)} samples in {elapsed:.1f}s")

    def _build_cache(self, cache_path, sentinel_path):
        # Remove corrupted cache if sentinel exists
        if os.path.exists(sentinel_path):
            if os.path.exists(cache_path):
                os.remove(cache_path)
            print(f"[CachedImageFolder] Found stale sentinel, rebuilding cache...")

        print(f"[CachedImageFolder] Building index for {self.root_dir} ...")
        # Create sentinel
        with open(sentinel_path, 'w') as f:
            f.write(str(os.getpid()))

        t0 = time.time()
        paths = []
        labels = []

        # Sorted class dirs for deterministic label assignment
        class_dirs = sorted(
            entry.name for entry in os.scandir(self.root_dir)
            if entry.is_dir() and not entry.name.startswith('_')
        )

        IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for label_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.root_dir, class_name)
            try:
                for fname in os.listdir(class_path):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in IMG_EXTENSIONS:
                        rel_path = os.path.join(class_name, fname)
                        paths.append(rel_path)
                        labels.append(label_idx)
            except OSError:
                continue

            if label_idx % 50000 == 0 and label_idx > 0:
                elapsed = time.time() - t0
                print(f"  ... scanned {label_idx}/{len(class_dirs)} classes, "
                      f"{len(paths)} images, {elapsed:.0f}s")

        elapsed = time.time() - t0
        print(f"[CachedImageFolder] Scan complete: {len(paths)} images, "
              f"{len(class_dirs)} classes, {elapsed:.0f}s")

        # Write parquet with zstd compression
        print(f"[CachedImageFolder] Writing parquet cache...")
        table = pa.table({
            'path': pa.array(paths, type=pa.string()),
            'label': pa.array(labels, type=pa.int32()),
        })
        pq.write_table(table, cache_path, compression='zstd')

        # Remove sentinel
        os.remove(sentinel_path)
        elapsed = time.time() - t0
        print(f"[CachedImageFolder] Cache saved to {cache_path} ({elapsed:.0f}s total)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        label = int(self.labels[idx])
        img_path = os.path.join(self.root_dir, rel_path)

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

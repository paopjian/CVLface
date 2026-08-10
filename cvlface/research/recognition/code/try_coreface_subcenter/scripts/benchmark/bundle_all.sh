#!/bin/bash
# Bundle all data formats for benchmarking.
# Usage: bash bundle_all.sh --source_dir /data1/dataset_0605/try --save_root /data1/dataset_0605/benchmark
set -e

SOURCE_DIR=""
SAVE_ROOT=""
QUALITY=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --source_dir) SOURCE_DIR="$2"; shift 2 ;;
        --save_root) SAVE_ROOT="$2"; shift 2 ;;
        --quality) QUALITY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$SOURCE_DIR" ] || [ -z "$SAVE_ROOT" ]; then
    echo "Usage: bash bundle_all.sh --source_dir <path> --save_root <path> [--quality 100]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Source: $SOURCE_DIR ==="
echo "=== Save root: $SAVE_ROOT ==="
echo ""

# ImageFolder: no bundling needed, just record the path
echo "[1/5] ImageFolder: no bundling needed (raw directory)"
echo ""

# LMDB
echo "[2/5] Bundling LMDB..."
python "$SCRIPT_DIR/bundle_lmdb.py" \
    --source_dir "$SOURCE_DIR" \
    --save_dir "$SAVE_ROOT/lmdb" \
    --quality "$QUALITY"
echo ""

# RecordIO
echo "[3/5] Bundling RecordIO..."
python "$SCRIPT_DIR/bundle_rec.py" \
    --source_dir "$SOURCE_DIR" \
    --save_dir "$SAVE_ROOT/recordio" \
    --quality "$QUALITY"
echo ""

# WebDataset
echo "[4/5] Bundling WebDataset..."
python "$SCRIPT_DIR/bundle_webdataset.py" \
    --source_dir "$SOURCE_DIR" \
    --save_dir "$SAVE_ROOT/webdataset" \
    --quality "$QUALITY"
echo ""

# HDF5
echo "[5/5] Bundling HDF5..."
python "$SCRIPT_DIR/bundle_hdf5.py" \
    --source_dir "$SOURCE_DIR" \
    --save_dir "$SAVE_ROOT/hdf5" \
    --quality "$QUALITY"
echo ""

# Zarr
echo "[6/6] Bundling Zarr..."
python "$SCRIPT_DIR/bundle_zarr.py" \
    --source_dir "$SOURCE_DIR" \
    --save_dir "$SAVE_ROOT/zarr" \
    --quality "$QUALITY"
echo ""

echo "=== All formats bundled ==="
echo "Disk usage:"
echo "  ImageFolder: $(du -sh "$SOURCE_DIR" | cut -f1)"
echo "  LMDB:        $(du -sh "$SAVE_ROOT/lmdb" | cut -f1)"
echo "  RecordIO:    $(du -sh "$SAVE_ROOT/recordio" | cut -f1)"
echo "  WebDataset:  $(du -sh "$SAVE_ROOT/webdataset" | cut -f1)"
echo "  HDF5:        $(du -sh "$SAVE_ROOT/hdf5" | cut -f1)"
echo "  Zarr:        $(du -sh "$SAVE_ROOT/zarr" | cut -f1)"

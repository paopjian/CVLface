#!/usr/bin/env python3
"""Reproduce TinyFace's legacy CPU-memory slowdown under controlled pressure.

The benchmark intentionally mirrors the legacy operations:
  label_mat = probe_labels[:, None] == gallery_labels[None, :]
  score_mat_m = score_mat[match_indices, :]
  label_mat_m = label_mat[match_indices, :]
  torch.gather(label_mat_m, ...)

This script is diagnostic only. It does not change system settings. Pressure
workers are terminated automatically and stop when available memory is low.
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import signal
import time

import numpy as np


PROBE_COUNT = 3728
GALLERY_COUNT = 157871
PAGE_SIZE = 4096


def available_gib():
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def swap_activity():
    values = {}
    with open("/proc/vmstat", encoding="ascii") as handle:
        for line in handle:
            key, value = line.split()[:2]
            if key in {"pswpin", "pswpout"}:
                values[key] = int(value)
    return values


def touch_buffer(buffer):
    view = memoryview(buffer).cast("B")
    for offset in range(0, len(view), PAGE_SIZE):
        view[offset] = (offset // PAGE_SIZE) & 0xFF


def numa_alloc(size_bytes, node):
    libnuma = ctypes.CDLL("libnuma.so.1")
    libnuma.numa_available.restype = ctypes.c_int
    if libnuma.numa_available() < 0:
        raise RuntimeError("libnuma reports that NUMA is unavailable")
    libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
    libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    pointer = libnuma.numa_alloc_onnode(size_bytes, node)
    if not pointer:
        raise MemoryError(f"numa_alloc_onnode failed for {size_bytes} bytes")
    array_type = ctypes.c_ubyte * size_bytes
    array = array_type.from_address(pointer)
    return libnuma, pointer, array


def pressure_worker(size_gib, node, bandwidth, stop_event):
    try:
        size_bytes = int(size_gib * (1024 ** 3))
        if node is None:
            buffer = bytearray(size_bytes)
            touch_buffer(buffer)
            allocated = buffer
            free_callback = lambda: None
        else:
            libnuma, pointer, array = numa_alloc(size_bytes, node)
            touch_buffer(array)
            allocated = array
            free_callback = lambda: libnuma.numa_free(pointer, size_bytes)

        if bandwidth:
            view = memoryview(allocated).cast("B")
            stride = PAGE_SIZE
            while not stop_event.is_set():
                checksum = 0
                for offset in range(0, len(view), stride):
                    checksum += view[offset]
                if checksum < 0:
                    raise AssertionError("unreachable")
        else:
            stop_event.wait()
        free_callback()
    except BaseException as error:
        print(f"pressure worker failed: {error}", flush=True)
        raise


def create_legacy_inputs():
    rng = np.random.default_rng(20260810)
    score_mat = rng.standard_normal((PROBE_COUNT, GALLERY_COUNT), dtype=np.float32)
    probe_labels = np.arange(PROBE_COUNT, dtype=np.int64)
    gallery_labels = np.arange(PROBE_COUNT, dtype=np.int64)
    gallery_labels = np.pad(
        gallery_labels,
        (0, GALLERY_COUNT - PROBE_COUNT),
        constant_values=-100,
    )
    label_mat = probe_labels[:, None] == gallery_labels[None, :]
    return score_mat, label_mat


def legacy_benchmark():
    import torch

    score_mat, label_mat = create_legacy_inputs()
    stage_start = time.perf_counter()
    match_indices = label_mat.astype(np.bool_).any(axis=1)
    score_mat_m = score_mat[match_indices, :]
    label_mat_m = label_mat[match_indices, :]
    split_elapsed = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    score_tensor = torch.from_numpy(score_mat_m)
    topk_indices = torch.topk(score_tensor, k=20, dim=1, sorted=True).indices
    label_tensor = torch.from_numpy(label_mat_m.astype(np.bool_))
    gathered = torch.gather(label_tensor, 1, topk_indices)
    gathered.sum().item()
    gather_elapsed = time.perf_counter() - stage_start
    print(
        f"legacy split={split_elapsed:.3f}s gather={gather_elapsed:.3f}s "
        f"rss_inputs={score_mat.nbytes / 1024**3:.2f}GiB+{label_mat.nbytes / 1024**3:.2f}GiB",
        flush=True,
    )
    del gathered, label_tensor, topk_indices, score_tensor
    del score_mat_m, label_mat_m, score_mat, label_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "occupancy", "bandwidth", "remote"), default="baseline")
    parser.add_argument("--memory-gib", type=float, default=16.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-available-gib", type=float, default=120.0)
    args = parser.parse_args()

    if args.memory_gib < 0 or args.duration <= 0:
        parser.error("memory-gib must be non-negative and duration must be positive")
    if args.mode == "remote" and args.workers != 2:
        parser.error("remote mode requires exactly two workers")
    if available_gib() - args.memory_gib < args.min_available_gib:
        parser.error(
            f"refusing to start: available={available_gib():.1f}GiB, "
            f"requested={args.memory_gib:.1f}GiB, reserve={args.min_available_gib:.1f}GiB"
        )

    print(
        f"mode={args.mode} memory={args.memory_gib:.1f}GiB workers={args.workers} "
        f"available_before={available_gib():.1f}GiB swap={swap_activity()}",
        flush=True,
    )
    stop_event = mp.Event()
    processes = []
    if args.mode != "baseline":
        per_worker = args.memory_gib / args.workers
        nodes = [0, 1] if args.mode == "remote" else [None] * args.workers
        for worker_id in range(args.workers):
            process = mp.Process(
                target=pressure_worker,
                args=(per_worker, nodes[worker_id], args.mode == "bandwidth", stop_event),
                daemon=True,
            )
            process.start()
            processes.append(process)
        time.sleep(3)
        print(f"available_after_pressure={available_gib():.1f}GiB", flush=True)

    try:
        legacy_benchmark()
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            if available_gib() < args.min_available_gib:
                print("stopping early: available memory reserve reached", flush=True)
                break
            time.sleep(1)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
                process.join()
        print(
            f"available_after_cleanup={available_gib():.1f}GiB swap={swap_activity()}",
            flush=True,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

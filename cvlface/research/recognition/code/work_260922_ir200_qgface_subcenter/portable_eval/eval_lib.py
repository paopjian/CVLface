#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可迁移评测工具库: 双流 nvjpeg / folder 读取 → TRT 推理 → NxN 匹配 → TPIR@FPIR.

设计原则:
  - 除 TRT engine 外零仓库依赖 (engine 支持从 ONNX 构建, 纯 TRT API);
  - NxN 匹配为纯 torch 实现 (fp16 GEMM + bincount 直方图, 多 GPU 线程),
    口径与 v7 一致 (hist_bins 可调), 速度慢于融合核但可移植;
  - 两条提特征管线: `nvjpeg` (GPU 批量解码 + 双流重叠) 与
    `cv2` (CPU 多线程解码, 任意环境兜底);
  - worker 均为模块级函数且配置经参数传递, 兼容 mp spawn。

依赖: torch / tensorrt / opencv-python (必须);
      nvidia-nvjpeg-cu12 (pip, 含头文件+库) + CUDA 头文件 (nvjpeg 管线必须)。
"""
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# ============================ 依赖发现 ============================


def _conda_prefix():
    return Path(sys.executable).parent.parent


def find_nvjpeg():
    """返回 (include_dir, lib_file); 找不到抛 FileNotFoundError."""
    cands_include, cands_lib = [], []
    if os.environ.get('NVJPEG_HOME'):
        cands_include.append(Path(os.environ['NVJPEG_HOME']) / 'include')
        cands_lib.append(Path(os.environ['NVJPEG_HOME']) / 'lib')
    try:
        import site
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            cands_include.append(Path(sp) / 'nvidia/nvjpeg/include')
            cands_lib.append(Path(sp) / 'nvidia/nvjpeg/lib')
    except Exception:
        pass
    cands_include += [_conda_prefix() / 'include', Path('/usr/local/cuda/include')]
    cands_lib += [_conda_prefix() / 'lib', Path('/usr/local/cuda/lib64')]

    for d in cands_include:
        if (d / 'nvjpeg.h').exists():
            for d2 in cands_lib:
                for name in ('libnvjpeg.so.12', 'libnvjpeg.so'):
                    if (d2 / name).exists():
                        return str(d), str(d2 / name)
            raise FileNotFoundError('找到 nvjpeg.h 但未找到 libnvjpeg (so.12/so)')
    raise FileNotFoundError('未找到 nvjpeg.h (pip install nvidia-nvjpeg-cu12 或设 NVJPEG_HOME)')


def find_cudart():
    """返回可直接链接的 libcudart 路径 (无则回退 -lcudart)."""
    import site
    cands = [_conda_prefix() / 'lib/libcudart.so']
    try:
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            cands.append(Path(sp) / 'nvidia/cuda_runtime/lib/libcudart.so.12')
            cands.append(Path(sp) / 'nvidia/cuda_runtime/lib/libcudart.so')
    except Exception:
        pass
    cands.append(Path('/usr/local/cuda/lib64/libcudart.so'))
    for c in cands:
        if c.exists():
            return str(c)
    return '-lcudart'


def find_cuda_include():
    cands = []
    if os.environ.get('CUDA_HOME'):
        cands.append(Path(os.environ['CUDA_HOME']) / 'include')
    cands += [Path('/usr/local/cuda/include'),
              _conda_prefix() / 'targets/x86_64-linux/include',
              _conda_prefix() / 'include']
    for d in cands:
        if (d / 'cuda_runtime.h').exists():
            return str(d)
    raise FileNotFoundError('未找到 cuda_runtime.h (装 CUDA toolkit 或设 CUDA_HOME)')


# ============================ nvjpeg 批量解码扩展 ============================

_NVJPEG_CPP = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <stdexcept>

static nvjpegHandle_t g_handle = nullptr;
static nvjpegJpegState_t g_state = nullptr;
static long g_init_batch = 0;
static long g_init_dev = -1;

void pe_nvjpeg_init(long device_id, long batch) {
    if (g_handle && g_init_batch == batch && g_init_dev == device_id) return;
    if (g_handle) { nvjpegDestroy(g_handle); g_handle = nullptr; }
    cudaSetDevice((int)device_id);
    if (nvjpegCreateSimple(&g_handle) != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("nvjpegCreateSimple failed");
    if (nvjpegJpegStateCreate(g_handle, &g_state) != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("nvjpegJpegStateCreate failed");
#if 1  // find_nvjpeg 优先 libnvjpeg.so.12, 其头文件即 12.x API。
       // 若确需 11.x, 把本分支换回第 5 参 device_id 的旧签名
    // 12.x: 第 5 参为输出格式 (RGBI 直出 HWC)
    if (nvjpegDecodeBatchedInitialize(g_handle, g_state, (int)batch, 1,
                                      NVJPEG_OUTPUT_RGBI) != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("DecodeBatchedInitialize failed");
#else
    // 11.x 及以下: 第 5 参为 device_id
    if (nvjpegDecodeBatchedInitialize(g_handle, g_state, (int)batch, 1,
                                      (int)device_id) != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("DecodeBatchedInitialize failed");
#endif
    g_init_batch = batch;
    g_init_dev = device_id;
}

// rec 读取: 按 offsets pread 整批记录 [8B 魔数+lrecord][24B 头][jpeg][pad]
// slot(0/1) 双缓冲 host 中转; sync=1 时返回即解码完成; stream_ptr 为解码流
void pe_decode_rec(long fd, torch::Tensor offsets, torch::Tensor out,
                   long jpeg_offset_in_record, long slot, long stream_ptr,
                   long sync) {
    long B = offsets.size(0);
    const int64_t* offs = offsets.data_ptr<int64_t>();
    unsigned char* outp = out.data_ptr<unsigned char>();
    static thread_local std::vector<unsigned char> buf[2];
    static thread_local std::vector<const unsigned char*> datas[2];
    static thread_local std::vector<size_t> sizes[2];
    static thread_local std::vector<nvjpegImage_t> dest[2];
    if (buf[slot].capacity() == 0) buf[slot].reserve(1 << 22);
    datas[slot].resize(B); sizes[slot].resize(B); dest[slot].resize(B);

    buf[slot].clear();
    unsigned char hdr[8];
    long pixel_bytes = out.size(1) * out.size(2) * out.size(3);
    for (long b = 0; b < B; b++) {
        if (pread(fd, hdr, 8, offs[b]) != 8)
            throw std::runtime_error("pread header failed");
        unsigned int magic, lr;
        std::memcpy(&magic, hdr, 4);
        std::memcpy(&lr, hdr + 4, 4);
        long length = lr & 0x1FFFFFFF;
        long at = buf[slot].size();
        buf[slot].resize(at + length);
        if (pread(fd, buf[slot].data() + at, length, offs[b] + 8) != length)
            throw std::runtime_error("pread record failed");
        datas[slot][b] = buf[slot].data() + at + jpeg_offset_in_record;
        sizes[slot][b] = length - jpeg_offset_in_record;
        std::memset(&dest[slot][b], 0, sizeof(nvjpegImage_t));
        dest[slot][b].channel[0] = outp + b * pixel_bytes;
        dest[slot][b].pitch[0] = out.size(2) * out.size(3);
    }
    nvjpegStatus_t st = nvjpegDecodeBatched(g_handle, g_state, datas[slot].data(),
                                            sizes[slot].data(), dest[slot].data(),
                                            (cudaStream_t)stream_ptr);
    if (st != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("nvjpegDecodeBatched failed: " + std::to_string((int)st));
    if (sync) cudaStreamSynchronize((cudaStream_t)stream_ptr);
}

// host 内存读取 (folder 场景): blobs 已拼接进 host_u8, data_offs/sizes 为切片
void pe_decode_host(torch::Tensor host_u8, torch::Tensor data_offs,
                    torch::Tensor sizes, torch::Tensor out,
                    long slot, long stream_ptr, long sync) {
    long B = out.size(0);
    unsigned char* base = host_u8.data_ptr<unsigned char>();
    const int64_t* offs = data_offs.data_ptr<int64_t>();
    const int64_t* szs = sizes.data_ptr<int64_t>();
    unsigned char* outp = out.data_ptr<unsigned char>();
    static thread_local std::vector<const unsigned char*> datas[2];
    static thread_local std::vector<size_t> sizes_v[2];
    static thread_local std::vector<nvjpegImage_t> dest[2];
    datas[slot].resize(B); sizes_v[slot].resize(B); dest[slot].resize(B);
    long pixel_bytes = out.size(1) * out.size(2) * out.size(3);
    for (long b = 0; b < B; b++) {
        datas[slot][b] = base + offs[b];
        sizes_v[slot][b] = (size_t)szs[b];
        std::memset(&dest[slot][b], 0, sizeof(nvjpegImage_t));
        dest[slot][b].channel[0] = outp + b * pixel_bytes;
        dest[slot][b].pitch[0] = out.size(2) * out.size(3);
    }
    nvjpegStatus_t st = nvjpegDecodeBatched(g_handle, g_state, datas[slot].data(),
                                            sizes_v[slot].data(), dest[slot].data(),
                                            (cudaStream_t)stream_ptr);
    if (st != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("nvjpegDecodeBatched failed: " + std::to_string((int)st));
    if (sync) cudaStreamSynchronize((cudaStream_t)stream_ptr);
}
"""

_NVJPEG_EXT = None


def build_nvjpeg_ext(force=False):
    """编译(带缓存) nvjpeg 批量解码扩展; 返回模块."""
    global _NVJPEG_EXT
    if _NVJPEG_EXT is not None and not force:
        return _NVJPEG_EXT
    # 用绝对路径调 python (未 conda activate) 时, pip 装的 ninja 不在子进程
    # PATH 里, load_inline 会报 "Ninja is required"; python 同级目录补进 PATH
    exe_dir = str(Path(sys.executable).parent)
    if exe_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load_inline
    inc_dir, lib_file = find_nvjpeg()
    # 预载头文件对应的 libnvjpeg: 系统库路径 (ld.so.cache) 里若有别的
    # libnvjpeg.so.12 (如 /usr/local/cuda-12.6 的 12.3.3), 会按 soname 抢先
    # 加载, 造成头文件/运行库版本错位 (2026-09-20 溯源, 见本目录 README)
    import ctypes
    ctypes.CDLL(lib_file, mode=ctypes.RTLD_GLOBAL)
    cuda_inc = find_cuda_include()
    cudart = find_cudart()
    _NVJPEG_EXT = load_inline(
        name='pe_nvjpeg_batched_v1',
        cpp_sources=_NVJPEG_CPP,
        functions=['pe_nvjpeg_init', 'pe_decode_rec', 'pe_decode_host'],
        extra_include_paths=[inc_dir, cuda_inc],
        extra_cflags=['-O3'],
        extra_ldflags=[lib_file, cudart],
        verbose=False,
    )
    return _NVJPEG_EXT


# ============================ TRT engine ============================

class TrtEngine:
    """TRT engine; 支持静态 batch 与动态 batch (ONNX batch 维为符号) engine.

    动态 engine 下 batch_size 参数 = 推理缓冲的批容量, 每次调用按实际 B
    set_input_shape; 静态 engine 行为不变。
    """

    def __init__(self, engine_path, batch_size, device=0):
        import tensorrt as trt
        self.batch_size = batch_size
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
        if self.engine is None:
            raise RuntimeError(f'engine 反序列化失败: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        assert self.input_name and self.output_name
        trt2torch = {trt.float16: torch.float16, trt.float32: torch.float32}
        self.input_dtype = trt2torch[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = trt2torch[self.engine.get_tensor_dtype(self.output_name)]
        self.output_dim = self.engine.get_tensor_shape(self.output_name)[-1]
        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.in_tail = in_shape[1:]
        self.dynamic = in_shape[0] == -1
        with torch.cuda.device(device):
            self.d_input = torch.zeros(batch_size, 3, 112, 112,
                                       dtype=self.input_dtype, device='cuda')
            self.d_output = torch.zeros(batch_size, self.output_dim,
                                        dtype=self.output_dtype, device='cuda')
            self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
            self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())
            self.stream = torch.cuda.current_stream()

    def infer(self, x):
        """x: CPU/GPU (B,3,112,112); 返回 fp32 (B,D) GPU 张量 (异步, 未同步)."""
        with torch.cuda.device(self.device):
            B = x.shape[0]
            assert B <= self.batch_size, f'B={B} 超过 engine 缓冲 {self.batch_size}'
            if self.dynamic:
                self.context.set_input_shape(self.input_name, (B,) + self.in_tail)
            if x.device.type == 'cuda':
                self.d_input[:B].copy_(x.to(self.input_dtype))
            else:
                self.d_input[:B].copy_(
                    x.to(self.input_dtype).cuda(non_blocking=True))
            self.context.execute_async_v3(self.stream.cuda_stream)
            return self.d_output[:B].float()


def build_engine_from_onnx(onnx_path, engine_path, batch_size, fp16=True,
                           workspace_gb=4, dynamic_max=0):
    """从 ONNX 构建 TRT engine (纯 TRT API, 不依赖训练框架).

    TRT 10.7+/11: 弱精度 FP16 flag 已移除, 用 STRONGLY_TYPED 网络,
    精度跟随 ONNX (fp16 导出即得 fp16 engine); fp16 参数仅为兼容旧签名。
    dynamic_max > 0 时把 batch 维改为符号构建动态 engine (min=64,
    opt=batch_size, max=dynamic_max); 否则静态 batch = ONNX 声明值
    (batch_size 参数不参与静态形状)。
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    model = None
    if dynamic_max > 0:
        import onnx
        model = onnx.load(str(onnx_path), load_external_data=False)
        for io in (*model.graph.input, *model.graph.output):
            d0 = io.type.tensor_type.shape.dim[0]
            d0.ClearField('dim_value')
            d0.dim_param = 'N'
        onnx.save(model, str(engine_path) + '.dyn.onnx')
        assert parser.parse_from_file(str(engine_path) + '.dyn.onnx'), \
            '动态 ONNX 解析失败: ' + str(parser.get_error(0))
    else:
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errs = '\n'.join(str(parser.get_error(i))
                                 for i in range(parser.num_errors))
                raise RuntimeError(f'ONNX 解析失败:\n{errs}')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if dynamic_max > 0:
        name = network.get_input(0).name
        tail = tuple(network.get_input(0).shape)[1:]
        profile = builder.create_optimization_profile()
        profile.set_shape(name, (64,) + tail, (batch_size,) + tail,
                          (dynamic_max,) + tail)
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TRT build 失败')
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    if model is not None:
        Path(str(engine_path) + '.dyn.onnx').unlink(missing_ok=True)
    return str(engine_path)


# ============================ 数据源 ============================

class FolderSource:
    """ImageFolder 布局: root/档案目录/图片; id = 目录名排序编号."""

    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

    def __init__(self, root):
        self.root = Path(root)
        self.paths, self.id_map = [], {}
        classes = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)))
        for ci, c in enumerate(classes):
            self.id_map[c] = ci
            cdir = os.path.join(root, c)
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith(self.IMG_EXTS):
                    self.paths.append(os.path.join(cdir, fname))
        self.ids = np.array([self.id_map[Path(p).parent.name] for p in self.paths],
                            dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def shards(self, world):
        """[(rank 的条目列表, rank 的 ids)], rank::world 分片, 覆盖全量."""
        return [(self.paths[r::world], self.ids[r::world]) for r in range(world)]

    @staticmethod
    def read_blobs(chunk_paths):
        """读一批文件 → (拼接 host 缓冲, 切片偏移, 切片长度) 供 nvjpeg host 解码."""
        blobs = [open(p, 'rb').read() for p in chunk_paths]
        buf = np.frombuffer(b''.join(blobs), dtype=np.uint8)
        sizes = np.array([len(b) for b in blobs], dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        return torch.from_numpy(buf.copy()), torch.from_numpy(offs), \
            torch.from_numpy(sizes)


class RecSource:
    """mxnet RecordIO 兼容 (bundle_images_into_rec_v2 格式): train.rec/idx; id 取 train.tsv 第 3 列.

    idx/tsv 的 Python 逐行 parse 在千万行级 rec 上每进程要 1-2 分钟,
    首次解析后落 train.offsets.npy / train.labels.npy, spawn 子进程直接加载.
    """

    def __init__(self, rec_dir, cache=True):
        self.rec_dir = Path(rec_dir)
        self.rec_path = str(self.rec_dir / 'train.rec')
        off_cache = self.rec_dir / 'train.offsets.npy'
        lab_cache = self.rec_dir / 'train.labels.npy'
        if cache and off_cache.exists() and lab_cache.exists():
            self.offsets = np.load(off_cache)
            self.ids = np.load(lab_cache)
            assert len(self.offsets) == len(self.ids), 'idx/tsv 缓存行数不一致'
        else:
            from array import array
            offs, labels = array('q'), array('q')
            with open(self.rec_dir / 'train.idx') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        offs.append(int(parts[1]))
            with open(self.rec_dir / 'train.tsv') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3:
                        labels.append(int(parts[2]))
            assert len(offs) == len(labels), \
                f'train.idx ({len(offs)}) 与 train.tsv ({len(labels)}) 行数不一致'
            self.offsets = np.frombuffer(offs, dtype=np.int64).copy()
            self.ids = np.frombuffer(labels, dtype=np.int64).copy()
            if cache:
                # tmp 名带 pid: 并发首开 (如 8 个评估 worker 同时解析同一 rec) 时
                # 固定名会互相 rename 掉对方的临时文件 → FileNotFoundError
                tmp = self.rec_dir / f'.offsets.tmp.{os.getpid()}.npy'
                np.save(tmp, self.offsets)
                os.replace(tmp, off_cache)
                tmp = self.rec_dir / f'.labels.tmp.{os.getpid()}.npy'
                np.save(tmp, self.ids)
                os.replace(tmp, lab_cache)
        self.fd = os.open(self.rec_path, os.O_RDONLY)

    def __len__(self):
        return len(self.offsets)

    def shards(self, world):
        return [(self.offsets[r::world], self.ids[r::world]) for r in range(world)]

    @staticmethod
    def read_records(fd, chunk_offsets):
        """读一批记录 → uint8 CHW 张量 (CPU, cv2 管线用)."""
        import cv2
        out = np.empty((len(chunk_offsets), 112, 112, 3), dtype=np.uint8)
        for j, off in enumerate(chunk_offsets):
            hdr = os.pread(fd, 8, off)
            _, lr = struct.unpack('<II', hdr)
            length = lr & 0x1FFFFFFF
            data = os.pread(fd, length, off + 8)
            img = cv2.imdecode(np.frombuffer(data[24:], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f'record@{off} 解码失败')
            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112))
            out[j] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(out)


# ============================ 特征提取 ============================

def _norm_cpu_to_fp16(x_u8_bchw, device):
    """uint8 BCHW → 归一化 fp16 GPU."""
    x = x_u8_bchw.to(device, non_blocking=True)
    return x.float().div_(255.0).sub_(0.5).div_(0.5).half()


def _nvjpeg_worker(rank, world, cfg, queue):
    """双流重叠 worker: 解码流 (nvjpeg batched 分片 + 归一化) / 推理流 (TRT).

    nvjpegDecodeBatched 对大批次总字节数有内部限制 (512×10.7KB 会段错误,
    256×10.7KB 正常), 故解码按 NVCHUNK 分片多次调用, 结果写入同批 fp16
    缓冲的不同行段, TRT 仍按整批推理; 解码与推理经双缓冲 + event 重叠。
    """
    torch.cuda.set_device(rank)
    ext = build_nvjpeg_ext()
    ext.pe_nvjpeg_init(rank, cfg['nvjpeg_chunk'])
    engine = TrtEngine(cfg['engine_path'], cfg['batch_size'], device=rank)
    infer_stream = engine.stream
    decode_stream = torch.cuda.Stream(device=rank)
    B, D = cfg['batch_size'], engine.output_dim
    NV = min(cfg['nvjpeg_chunk'], B)
    n_sub = (B + NV - 1) // NV
    dev = f'cuda:{rank}'

    src = RecSource(cfg['source']) if cfg['source_type'] == 'rec' \
        else FolderSource(cfg['source'])
    fd = src.fd if cfg['source_type'] == 'rec' else -1
    chunk_items, chunk_ids = src.shards(world)[rank]
    n_full = len(chunk_items) // B
    if cfg.get('max_batches'):
        n_full = min(n_full, cfg['max_batches'])

    u8 = [torch.empty(NV, 112, 112, 3, dtype=torch.uint8, device=dev)
          for _ in range(2)]
    buf_in = [torch.empty(B, 3, 112, 112, dtype=torch.float16, device=dev)
              for _ in range(2)]
    dec_done = [torch.cuda.Event() for _ in range(2)]   # 按 sub 调用轮转
    inf_done = [torch.cuda.Event() for _ in range(2)]   # 按 batch 轮转
    for e in inf_done:
        e.record(infer_stream)
    CHUNK_B = 64  # 每 64 批才做一次 D2H 回拷, 避免逐批同步破坏流水线重叠
    buf_gpu = torch.empty(min(CHUNK_B, n_full) * B, D, dtype=torch.float16,
                          device=dev)
    cpu_chunks = []

    def enqueue(i):
        bslot = i % 2
        decode_stream.wait_event(inf_done[bslot])   # buf_in 上轮推理已完成
        items = chunk_items[i * B:(i + 1) * B]
        for j in range(n_sub):
            sslot = (i * n_sub + j) % 2             # sub 调用级双缓冲 (host 中转)
            dec_done[sslot].synchronize()           # host 等同 slot 上轮 H2D+解码完成
            with torch.cuda.stream(decode_stream):
                if cfg['source_type'] == 'rec':
                    seg = torch.tensor(items[j * NV:(j + 1) * NV],
                                       dtype=torch.int64)
                    ext.pe_decode_rec(fd, seg, u8[sslot], 24, sslot,
                                      decode_stream.cuda_stream, 0)
                else:
                    # encoded 数据留在 host (nvjpeg 内部自行 H2D);
                    # offs/sizes 是 host 指针表, .to(dev) 会导致 C++ 侧解引用显存地址
                    buf, offs, sizes = FolderSource.read_blobs(
                        items[j * NV:(j + 1) * NV])
                    ext.pe_decode_host(buf, offs, sizes, u8[sslot], sslot,
                                       decode_stream.cuda_stream, 0)
                x = u8[sslot].permute(0, 3, 1, 2)
                buf_in[bslot][j * NV:(j + 1) * NV].copy_(
                    _norm_cpu_to_fp16(x, dev))
                dec_done[sslot].record(decode_stream)
        infer_stream.wait_event(dec_done[(i * n_sub + n_sub - 1) % 2])
        out = engine.infer(buf_in[bslot])
        k = i % CHUNK_B
        buf_gpu[k * B:(k + 1) * B] = out.half()      # D2D, 不打断流水线
        inf_done[bslot].record(infer_stream)
        if k == CHUNK_B - 1 or i == n_full - 1:
            cpu_chunks.append(buf_gpu[:(k + 1) * B].cpu())  # 攒批回拷

    t0 = time.time()
    for i in range(n_full):
        enqueue(i)
    infer_stream.synchronize()
    decode_stream.synchronize()
    wall = time.time() - t0
    feats_shard = torch.cat(cpu_chunks)

    out_path = Path(cfg['tmp_dir']) / f'feats_rank{rank}.pt'
    torch.save({'feats': feats_shard, 'ids': chunk_ids[:n_full * B],
                'time': wall, 'n_images': n_full * B}, out_path)
    queue.put({'rank': rank, 'n_images': n_full * B, 'wall': wall,
               'path': str(out_path)})


def _cv2_worker(rank, world, cfg, queue):
    """CPU 多线程 cv2 解码 worker (兜底管线)."""
    import cv2
    cv2.setNumThreads(1)
    torch.cuda.set_device(rank)
    engine = TrtEngine(cfg['engine_path'], cfg['batch_size'], device=rank)
    B, D = cfg['batch_size'], engine.output_dim
    dev = f'cuda:{rank}'
    pool = ThreadPoolExecutor(max_workers=cfg['decode_threads'])

    src = RecSource(cfg['source']) if cfg['source_type'] == 'rec' \
        else FolderSource(cfg['source'])
    fd = src.fd if cfg['source_type'] == 'rec' else -1
    chunk_items, chunk_ids = src.shards(world)[rank]
    n_full = len(chunk_items) // B
    if cfg.get('max_batches'):
        n_full = min(n_full, cfg['max_batches'])
    CHUNK_B = 64
    buf_gpu = torch.empty(min(CHUNK_B, n_full) * B, D, dtype=torch.float16,
                          device=dev)
    cpu_chunks = []

    def decode_chunk(items):
        if cfg['source_type'] == 'rec':
            return RecSource.read_records(fd, items)
        blobs = [open(p, 'rb').read() for p in items]
        out = np.empty((len(blobs), 112, 112, 3), dtype=np.uint8)
        for j, b in enumerate(blobs):
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError('图片解码失败')
            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112))
            out[j] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(out)

    t0 = time.time()
    for i in range(n_full):
        items = chunk_items[i * B:(i + 1) * B]
        x_u8 = pool.submit(decode_chunk, items).result()
        with torch.cuda.device(rank):
            x = _norm_cpu_to_fp16(x_u8_bchw=x_u8, device=f'cuda:{rank}')
            out = engine.infer(x.permute(0, 3, 1, 2))
            k = i % CHUNK_B
            buf_gpu[k * B:(k + 1) * B] = out.half()
            if k == CHUNK_B - 1 or i == n_full - 1:
                cpu_chunks.append(buf_gpu[:(k + 1) * B].cpu())
    wall = time.time() - t0
    feats_shard = torch.cat(cpu_chunks)

    out_path = Path(cfg['tmp_dir']) / f'feats_rank{rank}.pt'
    torch.save({'feats': feats_shard, 'ids': chunk_ids[:n_full * B],
                'time': wall, 'n_images': n_full * B}, out_path)
    queue.put({'rank': rank, 'n_images': n_full * B, 'wall': wall,
               'path': str(out_path)})


def extract_features(source_type, source, engine_path, out_dir,
                     num_gpus=1, batch_size=512, pipeline='nvjpeg',
                     decode_threads=4, nvjpeg_chunk=256):
    """全量提特征 (跳过末尾不满批): 返回 (feats fp16 np 已归一化, ids np, meta).

    行号与 source 全量顺序一致 (rank::world 分片后按 rank 拼回).
    """
    if source_type == 'rec':
        n_all = len(RecSource(source))
    else:
        n_all = len(FolderSource(source))
    per_gpu_batches = (n_all // num_gpus) // batch_size
    n_kept = per_gpu_batches * batch_size * num_gpus
    if n_kept == 0:
        raise RuntimeError(
            f'数据 ({n_all} 张) 不足一个整批 (bs={batch_size} × {num_gpus} 卡) —— '
            f'减小 --bs 或增加数据量')
    if n_kept < n_all:
        print(f'[提示] 尾部不足整批的 {n_all - n_kept} 张被跳过 '
              f'(每卡 {per_gpu_batches} 批 × {num_gpus} 卡 × {batch_size})')

    tmp_dir = Path(out_dir) / '_tmp_feats'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg = {'source_type': source_type, 'source': str(source),
           'engine_path': str(engine_path), 'batch_size': batch_size,
           'decode_threads': decode_threads, 'tmp_dir': str(tmp_dir),
           'nvjpeg_chunk': nvjpeg_chunk}
    target = _nvjpeg_worker if pipeline == 'nvjpeg' else _cv2_worker
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(r, num_gpus, cfg, q))
             for r in range(num_gpus)]
    for p in procs:
        p.start()
    rows = []
    while len(rows) < num_gpus:
        try:
            rows.append(q.get(timeout=900))
        except Exception:
            break
    for p in procs:
        p.join(timeout=120)
    failed = [i for i, p in enumerate(procs) if p.exitcode not in (0, None)]
    if failed or len(rows) < num_gpus:
        raise RuntimeError(f'提特征 worker 失败: failed={failed}, '
                           f'收到 {len(rows)}/{num_gpus} (详见上方 traceback)')

    shards = [torch.load(Path(tmp_dir) / f'feats_rank{r}.pt',
                         map_location='cpu', weights_only=False)
              for r in range(num_gpus)]
    dim = shards[0]['feats'].shape[1]
    feats = np.empty((n_kept, dim), dtype=np.float16)
    ids_parts, row = [], 0
    for s in shards:
        ft = s['feats']
        ids_parts.append(s['ids'])
        # L2 归一化: fp32 逐块算、直接落 fp16 (数值与整块 fp32 路径逐位一致,
        # 免去整库 fp32 物化: 17.9M 时省 34.6GB 中间内存与两趟转换)
        for i in range(0, ft.shape[0], 1_000_000):
            blk = ft[i:i + 1_000_000].float()
            norm = blk.norm(dim=1).clamp_min(1e-12)
            out = (blk / norm.unsqueeze(1)).half()
            feats[row:row + out.shape[0]] = out.numpy()
            row += out.shape[0]
    ids = np.concatenate(ids_parts)
    total_time = max(s['time'] for s in shards)  # 墙钟 = 最慢卡

    meta = {'pipeline': pipeline, 'source_type': source_type,
            'n_images': int(feats.shape[0]), 'n_skipped_tail': n_all - n_kept,
            'num_gpus': num_gpus, 'batch_size': batch_size,
            'extract_time_s': round(total_time, 1),
            'throughput_img_s': round(feats.shape[0] / max(total_time, 1e-9), 1)}
    shutil.rmtree(tmp_dir, ignore_errors=True)  # 整目录清(残留分片一并清)
    return feats, ids, meta


# ============================ NxN 匹配 (纯 torch, 多 GPU) ============================

_IDX_DTYPE = None  # CUDA bincount 对 int32 的支持按 torch 版本而定, 首次调用探测
def pos_neg_hist_torch(feats, ids, num_gpus=1, block=8192, bins=2000,
                       lo=-1.0, hi=1.0):
    """全对 (NxN) 相似度直方图: pos=同 id, neg=跨 id, 严格上三角每对计一次.

    feats: fp32/fp16 np [N,D] 已 L2 归一化; fp16 连续输入零拷贝直用.
    ids: int64 np. 返回 (pos_hist, neg_hist) int64 np [bins].
    """
    global _IDX_DTYPE
    if feats.dtype == np.float16:
        f = torch.from_numpy(np.ascontiguousarray(feats))
    else:
        f = torch.from_numpy(np.ascontiguousarray(feats.astype(np.float16)))
    ids_t = torch.from_numpy(ids.astype(np.int64))
    N = f.shape[0]
    nb = (N + block - 1) // block
    gpus = min(num_gpus, torch.cuda.device_count())
    assert gpus >= 1, '无可用 GPU'
    if _IDX_DTYPE is None:
        try:
            torch.zeros(4, dtype=torch.int32,
                        device='cuda:0').bincount(minlength=2)
            _IDX_DTYPE = torch.int32
        except RuntimeError:
            _IDX_DTYPE = torch.int64

    # 每 GPU: 列分片常驻 + 全量 ids 常驻 (行块按 tile 流式 H2D)
    col_shards, ids_gpus = [], []
    for g in range(gpus):
        with torch.cuda.device(g):
            blocks = {bj: f[bj * block:min((bj + 1) * block, N)]
                      .to(f'cuda:{g}') for bj in range(g, nb, gpus)}
            col_shards.append(blocks)
            ids_gpus.append(ids_t.to(f'cuda:{g}'))
    torch.cuda.synchronize()

    pos_t = [torch.zeros(bins, dtype=torch.int64, device=f'cuda:{g}')
             for g in range(gpus)]
    neg_t = [torch.zeros(bins, dtype=torch.int64, device=f'cuda:{g}')
             for g in range(gpus)]
    scale = (bins - 1) / (hi - lo)

    def gpu_worker(g):
        blocks = col_shards[g]
        ids_gpu = ids_gpus[g]
        with torch.cuda.device(g):
            # A 块只随 bi 变化, 提到 bj 循环外: 否则非常驻 A 按 tile 数重复 H2D
            # (17.9M 规模 nb=2185 时 H2D 流量差 nb/7 倍)
            for bi in range(nb):
                r0, r1 = bi * block, min((bi + 1) * block, N)
                A = blocks[bi] if bi in blocks else \
                    f[r0:r1].to(f'cuda:{g}', non_blocking=False)
                for bj in range(bi, nb):
                    if bj % gpus != g:
                        continue
                    c0, c1 = bj * block, min((bj + 1) * block, N)
                    Bm = blocks[bj] if bj in blocks else \
                        f[c0:c1].to(f'cuda:{g}', non_blocking=False)
                    sim = (A @ Bm.T).float()
                    eq = ids_gpu[r0:r1, None] == ids_gpu[None, c0:c1]
                    idx = ((sim - lo) * scale).clamp_(0, bins - 1) \
                        .to(_IDX_DTYPE)  # int32 显存流量减半
                    if bi == bj:
                        tri = torch.ones_like(sim, dtype=torch.bool).triu_(1)
                        idx_f = idx[tri]
                        idx_p = idx[tri & eq]
                    else:
                        idx_f = idx.flatten()
                        idx_p = idx[eq]
                    # pos/neg 各计一次 bincount (idx_p 只算一遍, 原实现算了三遍)
                    cnt_p = torch.bincount(idx_p, minlength=bins)[:bins]
                    pos_t[g] += cnt_p
                    neg_t[g] += torch.bincount(idx_f, minlength=bins)[:bins] \
                        - cnt_p
            torch.cuda.synchronize()

    threads = [threading.Thread(target=gpu_worker, args=(g,)) for g in range(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    pos = pos_t[0].cpu().numpy().copy()
    neg = neg_t[0].cpu().numpy().copy()
    for g in range(1, gpus):
        pos += pos_t[g].cpu().numpy()
        neg += neg_t[g].cpu().numpy()
    expected = N * (N - 1) // 2
    assert int(pos.sum()) + int(neg.sum()) == expected, \
        f'对数不闭合: {int(pos.sum())}+{int(neg.sum())} != {expected}'
    return pos, neg


# ============ NxN 匹配 (融合核快速路径, 源自生产 v7 cuda_histpn_he_fused) ============

_FUSED_CU_PATH = Path(__file__).parent / 'cuda_histpn_he_fused.cu'
_fused_ext, _fused_tried = None, False


def _get_fused_ext():
    """懒编译融合核 (fp16 sim 直读 + full/pos 双桶直方图, 一次读全做完).

    编译成功后走 torch extensions 缓存; 任何失败返回 None (调用方回退 torch)。
    """
    global _fused_ext, _fused_tried
    if _fused_tried:
        return _fused_ext
    _fused_tried = True
    if not _FUSED_CU_PATH.exists():
        print(f'[匹配] 融合核源码缺失 ({_FUSED_CU_PATH.name}), 用 torch 实现')
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        exe_dir = str(Path(sys.executable).parent)
        if exe_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')
        cpp = ("void fused_he_pn(torch::Tensor x, long C, torch::Tensor rcd, "
               "torch::Tensor ccd, double lo, double hi, double invw, long bins, "
               "double thr_neg, double thr_pos, long is_diag, long row_off, "
               "long col_off, long max_pairs, torch::Tensor out_full, "
               "torch::Tensor out_pos, torch::Tensor ni, torch::Tensor nj, "
               "torch::Tensor ns, torch::Tensor ncnt, torch::Tensor pi, "
               "torch::Tensor pj, torch::Tensor ps, torch::Tensor pcnt, "
               "torch::Tensor gff, torch::Tensor gfp, double fine_lo, "
               "long fine_off0, long k_fine, long fine_bins, "
               "long blocks, long threads);")
        _fused_ext = load_inline(name='pe_fused_he_pn_v2', cpp_sources=cpp,
                                 cuda_sources=_FUSED_CU_PATH.read_text(),
                                 functions=['fused_he_pn'], verbose=False)
    except Exception as e:  # noqa: BLE001
        print(f'[匹配] 融合核编译失败 ({type(e).__name__}: {e}), 用 torch 实现')
        _fused_ext = None
    return _fused_ext


def pos_neg_hist_fused(feats, ids, num_gpus=1, block=16384, bins=2000,
                       lo=-1.0, hi=1.0, fine_lo=None, fine_bins=3_500_000,
                       return_fine=False):
    """全对直方图 · 融合核路径 (v7 同款核, 大 N 时比 torch 版快 ~5x).

    torch GEMM (fp16) 后单遍 kernel 直读 fp16 sim, shared-memory 双桶
    (full/pos) 原子聚合, 免 fp32 物化与 int64 索引; 阈值置界外即纯直方图模式。
    口径与 pos_neg_hist_torch 一致 (严格上三角每对计一次); 差异仅两点:
    1) 分块形状不同 (默认 16384 vs 8192), GEMM 归约顺序使个别边界对挪 bin
       (~0.5% 对数, bin 宽 1e-3), TPIR 影响可忽略;
    2) fp16 舍入致 |sim| 超出 [lo,hi] 的对被丢弃 (torch 版 clamp 进端点 bin),
       守恒校验允许 ≤0.001% 误差。
    fine_lo/fine_bins: 高端细直方图 (global atomic, 仅 v>=fine_lo 的对触发,
    bin 宽 (1-fine_lo)/fine_bins ~2e-7), 供 FPIR<=1e-9 等高分辨率档位;
    fine_lo=None 时关闭 (不分配, 行为与 v1 完全一致)。
    return_fine=True 时返回 (pos, neg, fine_pos, fine_neg)。
    编译失败自动回退 pos_neg_hist_torch (回退时 return_fine=True 会抛错)。
    """
    import queue
    ext = _get_fused_ext()
    if ext is None:
        if return_fine:
            raise RuntimeError('融合核不可用, 高分辨率直方图无回退实现')
        return pos_neg_hist_torch(feats, ids, num_gpus=num_gpus, block=block,
                                  bins=bins, lo=lo, hi=hi)
    if fine_lo is None:
        fine_lo, use_fine = hi + 1.0, False   # 阈值置界外 = 关闭细桶
    else:
        use_fine = True
    bin_w = (hi - lo) / bins
    fine_off0 = round((fine_lo - lo) / bin_w)   # 须对齐粗 bin 边界
    k_fine = round(fine_bins / (bins - fine_off0))  # 每粗 bin 的细 bin 数
    assert not use_fine or abs(fine_lo - (lo + fine_off0 * bin_w)) < 1e-9, \
        'fine_lo 未对齐粗 bin 边界'

    f = torch.from_numpy(np.ascontiguousarray(
        feats if feats.dtype == np.float16 else feats.astype(np.float16)))
    try:  # 锁页让 H2D 走 DMA (文件backed 内存注册失败则退回普通拷贝)
        torch.cuda.cudart().cudaHostRegister(f.data_ptr(), f.nbytes, 0)
    except Exception:
        f = f.pin_memory()
    codes = np.unique(ids, return_inverse=True)[1].astype(np.int32)
    code_t = torch.from_numpy(codes).pin_memory()
    N = f.shape[0]
    nb = (N + block - 1) // block
    gpus = min(num_gpus, torch.cuda.device_count())
    assert gpus >= 1, '无可用 GPU'

    invw = bins / (hi - lo)
    thr_n, thr_p = hi + 1.0, lo - 1.0  # 直方图模式: 阈值在界外, 不收集样本对
    CAP = 1024
    tiles_q = [queue.Queue() for _ in range(gpus)]
    for bi in range(nb):
        for bj in range(bi, nb):
            tiles_q[bj % gpus].put((bi, bj))
    for q in tiles_q:
        q.put(None)

    full_t, pos_t, valid_n = [None] * gpus, [None] * gpus, [0] * gpus
    fine_full_t = [None] * gpus if use_fine else None
    fine_pos_t = [None] * gpus if use_fine else None

    def worker(g):
        with torch.cuda.device(g):
            dev = torch.device(f'cuda:{g}')
            shard = {}
            for bj in range(g, nb, gpus):
                c0, c1 = bj * block, min((bj + 1) * block, N)
                shard[bj] = (f[c0:c1].to(dev, non_blocking=True),
                             code_t[c0:c1].to(dev, non_blocking=True))
            torch.cuda.current_stream().synchronize()
            accf = torch.zeros(bins, dtype=torch.int64, device=dev)
            accp = torch.zeros(bins, dtype=torch.int64, device=dev)
            outf = torch.zeros(bins, dtype=torch.int32, device=dev)
            outp = torch.zeros(bins, dtype=torch.int32, device=dev)
            if use_fine:
                accgf = torch.zeros(fine_bins, dtype=torch.int64, device=dev)
                accgp = torch.zeros(fine_bins, dtype=torch.int64, device=dev)
                gff = torch.zeros(fine_bins, dtype=torch.int32, device=dev)
                gfp = torch.zeros(fine_bins, dtype=torch.int32, device=dev)
            zi = lambda: torch.zeros(CAP, dtype=torch.int32, device=dev)
            zf = lambda: torch.zeros(CAP, dtype=torch.float32, device=dev)
            z1 = lambda: torch.zeros(1, dtype=torch.int32, device=dev)
            ni, nj, ns, ncnt = zi(), zi(), zf(), z1()
            pi, pj, ps, pcnt = zi(), zi(), zf(), z1()
            cache = (-1, None, None)
            valid = 0
            while True:
                it = tiles_q[g].get()
                if it is None:
                    break
                bi, bj = it
                r0, r1 = bi * block, min((bi + 1) * block, N)
                c0, c1 = bj * block, min((bj + 1) * block, N)
                diag = bi == bj
                if diag:
                    blk1, cd1 = shard[bi]
                    blk2, cd2 = blk1, cd1
                else:
                    if cache[0] == bi:  # 同一行块连续出队, 缓存复用免重复 H2D
                        blk1, cd1 = cache[1], cache[2]
                    else:
                        blk1 = f[r0:r1].to(dev, non_blocking=True)
                        cd1 = code_t[r0:r1].to(dev, non_blocking=True)
                        cache = (bi, blk1, cd1)
                    blk2, cd2 = shard[bj]
                sim = torch.matmul(blk1, blk2.T)  # fp16 GEMM
                outf.zero_(); outp.zero_(); ncnt.zero_(); pcnt.zero_()
                if use_fine:
                    gff.zero_(); gfp.zero_()
                ext.fused_he_pn(sim, c1 - c0, cd1, cd2, lo, hi, invw, bins,
                                thr_n, thr_p, int(diag), r0, c0, CAP,
                                outf, outp, ni, nj, ns, ncnt,
                                pi, pj, ps, pcnt,
                                gff if use_fine else outf,
                                gfp if use_fine else outp,
                                fine_lo, fine_off0, k_fine, fine_bins,
                                4096, 128)
                accf.add_(outf.to(torch.int64))
                accp.add_(outp.to(torch.int64))
                if use_fine:
                    accgf.add_(gff.to(torch.int64))
                    accgp.add_(gfp.to(torch.int64))
                R, C = r1 - r0, c1 - c0
                valid += R * (C - 1) // 2 if diag else R * C
            full_t[g] = accf.cpu()
            pos_t[g] = accp.cpu()
            if use_fine:
                fine_full_t[g] = accgf.cpu()
                fine_pos_t[g] = accgp.cpu()
            valid_n[g] = valid

    ths = [threading.Thread(target=worker, args=(g,)) for g in range(gpus)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    try:
        torch.cuda.cudart().cudaHostUnregister(f.data_ptr())
    except Exception:
        pass

    full = full_t[0].clone()
    pos = pos_t[0].clone()
    for g in range(1, gpus):
        full += full_t[g]
        pos += pos_t[g]
    neg = full - pos
    expected = N * (N - 1) // 2
    dropped = expected - int(full.sum())
    assert dropped <= expected // 100_000, \
        f'融合核守恒失败: 界外丢弃 {dropped}/{expected} 对 ' \
        f'(检查归一化与 hist_range)'
    if not return_fine:
        return pos.numpy(), neg.numpy()
    ffull = fine_full_t[0].clone()
    fpos = fine_pos_t[0].clone()
    for g in range(1, gpus):
        ffull += fine_full_t[g]
        fpos += fine_pos_t[g]
    fneg = ffull - fpos
    # 一致性校验: fine 编号由 coarse 位置推导 (kernel 内共享中间量),
    # 聚合回 coarse 段应逐位一致
    n_c = bins - fine_off0
    assert n_c * k_fine == fine_bins, 'fine_bins 与 (bins-off0)*k_fine 不等'
    fine_agg = ffull.view(n_c, k_fine).sum(1)
    coarse_seg = full[fine_off0:bins].to(torch.int64)
    total_diff = int((fine_agg - coarse_seg).abs().sum())
    if total_diff != 0:
        print(f'[警告] fine/coarse 聚合差 {total_diff} 对 (应=0)')
    return pos.numpy(), neg.numpy(), fpos.numpy(), fneg.numpy()


# ============================ TPIR@FPIR ============================

DEFAULT_FPIRS = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
HIRES_FPIRS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]


def tpir_from_hist(pos_hist, neg_hist, bins, lo=-1.0, hi=1.0, fpirs=None):
    """直方图 → TPIR@FPIR (%). 返回 (results, thresholds, (total_pos, total_neg))."""
    fpirs = fpirs or DEFAULT_FPIRS
    total_neg = int(neg_hist.sum())
    total_pos = int(pos_hist.sum())
    neg_cum = np.cumsum(neg_hist[::-1])
    pos_cum = np.cumsum(pos_hist[::-1])
    edges = np.linspace(lo, hi, bins + 1)
    results, thresholds = {}, {}
    for fpir in fpirs:
        target = fpir * total_neg
        idx = np.searchsorted(neg_cum, target, side='left')
        if idx >= bins:
            thr, tpir = lo, 1.0
        else:
            thr = float(edges[bins - 1 - idx])
            tpir = float(pos_cum[idx]) / total_pos if total_pos else 0.0
        results[f'{fpir:.0e}'] = tpir * 100
        thresholds[f'{fpir:.0e}'] = thr
    return results, thresholds, (total_pos, total_neg)


def tpir_from_hist_hires(coarse_pos, coarse_neg, fine_pos, fine_neg,
                         bins, lo=-1.0, hi=1.0, fine_lo=0.30,
                         fine_bins=3_500_000, fpirs=None):
    """粗+细双直方图 → TPIR@FPIR (%) 1e-4 ~ 1e-12.

    coarse (bins, [-1,1]) 只在阈值落入 [lo, fine_lo) 时使用 (1e-4~1e-8 档);
    fine (fine_bins, [fine_lo,hi], bin 宽 ~2e-7) 覆盖高阈值档 (1e-8~1e-12,
    实际取两者阈值更低者). 每档标注来源段. 分辨率: fpir 档目标对数 /
    (bin 宽) 须 >=1 才有定位意义, 小集 1e-11/1e-12 仍会触底 (results 带
    resolution_ok 标记).
    """
    fpirs = fpirs or HIRES_FPIRS
    total_pos = int(coarse_pos.sum())
    total_neg = int(coarse_neg.sum())
    assert int(fine_pos.sum()) <= total_pos and \
        int(fine_neg.sum()) <= total_neg, 'fine 直方图计数超出粗直方图'
    neg_cum = np.cumsum(coarse_neg[::-1])
    pos_cum = np.cumsum(coarse_pos[::-1])
    edges = np.linspace(lo, hi, bins + 1)
    fine_neg_cum = np.cumsum(fine_neg[::-1])
    fine_pos_cum = np.cumsum(fine_pos[::-1])
    fine_w = (hi - fine_lo) / fine_bins
    fine_edges = np.linspace(fine_lo, hi, fine_bins + 1)

    results, thresholds, source_seg, resolution = {}, {}, {}, {}
    for fpir in fpirs:
        target = fpir * total_neg
        # fine 段: 目标对数是否在 fine 覆盖范围内可达
        idx_f = np.searchsorted(fine_neg_cum, target, side='left')
        if idx_f < fine_bins:
            thr = float(fine_edges[fine_bins - 1 - idx_f])
            tpir = float(fine_pos_cum[idx_f]) / total_pos if total_pos else 0.0
            seg = 'fine'
            res_ok = target >= 1
        else:
            # fine 段全部负对仍不足 target (或 target=0) → 用粗段
            idx = np.searchsorted(neg_cum, target, side='left')
            if idx >= bins:
                thr, tpir = lo, 1.0
            else:
                thr = float(edges[bins - 1 - idx])
                tpir = float(pos_cum[idx]) / total_pos if total_pos else 0.0
            seg = 'coarse'
            res_ok = target >= (hi - lo) / bins
        results[f'{fpir:.0e}'] = tpir * 100
        thresholds[f'{fpir:.0e}'] = thr
        source_seg[f'{fpir:.0e}'] = seg
        resolution[f'{fpir:.0e}'] = bool(res_ok)
    return results, thresholds, (total_pos, total_neg), source_seg, resolution


# ============================ 特征缓存 ============================

def save_feature_cache(out_dir, tag, feats, ids, meta):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    arr = feats if feats.dtype == np.float16 else feats.astype(np.float16)
    np.save(d / f'{tag}.feats.fp16.npy', arr)
    np.save(d / f'{tag}.ids.npy', ids)
    with open(d / f'{tag}.meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return d / f'{tag}.feats.fp16.npy'


def load_feature_cache(out_dir, tag):
    d = Path(out_dir)
    # mmap_mode='c': COW 映射, 按页进内存且不写盘; 返回 fp16, 供匹配零拷贝直用
    feats = np.load(d / f'{tag}.feats.fp16.npy', mmap_mode='c')
    ids = np.load(d / f'{tag}.ids.npy')
    with open(d / f'{tag}.meta.json') as f:
        meta = json.load(f)
    return feats, ids, meta

"""
bench_mp_vs_mt_trt.py - TRT 7卡推理: 多进程 vs 多线程 启动开销/吞吐对比

目的: 评估"用多线程代替多进程分发到 7 卡跑 TensorRT"能否省掉启动损害。

测量 (两种方式各自):
  A. 端到端真实数据评估 (/data1/dataset_0605/try, ImageFolder)
     - 启动延迟 (worker 入口 - 主进程分发点; 多进程含 spawn+import)
     - engine 初始化 (deserialize engine + create context + 分配 buffer)
     - 推理 (DataLoader 循环, 含数据 IO + flip + infer)
     - 端到端 wall-clock
  B. 纯推理 micro-bench (固定随机 batch 重复推理, 隔离数据 IO)
     - 直接对比 GPU 并行推理吞吐, 检验多线程是否被 GIL 拖慢

engine 预先构建一次, 两种方式复用同一份 .engine 文件 (公平对比)。
特征不落盘 (聚焦启动+推理速度)。
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)

import os, sys, time, warnings, argparse, threading, gc
WORK_DIR = '/root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_0605'
sys.path.insert(0, WORK_DIR)
warnings.filterwarnings("ignore")


import numpy as np
np.bool = np.bool_

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

BATCH_SIZE = 256
NUM_WORKERS = 4
DATA_PATH = '/data1/dataset_0605/try'
CKPT_PATH = '/data2/dataset_0605/train_output/s2_body36_0605_06-10_2/checkpoints_every_epoch/epoch:14_step:135795'
ENGINE_PATH = '/tmp/trt_bench_mpmt/model_fp16.engine'


def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _collate(examples):
    pixel_values = torch.stack([e[0] for e in examples])
    return {"pixel_values": pixel_values}


class TRTInfer:
    """TRT FP16 推理器 (IO 按 tensor_mode 识别)"""
    def __init__(self, engine_path, batch_size=256):
        import tensorrt as trt
        self.batch_size = batch_size
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(memoryview(f.read()))
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = None, None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.input_name = n
            else:
                self.output_name = n
        self.d_input = torch.zeros(batch_size, 3, 112, 112, dtype=torch.float16, device='cuda')
        self.d_output = torch.zeros(batch_size, 512, dtype=torch.float16, device='cuda')
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())
        self.stream = torch.cuda.current_stream()

    def __call__(self, x):
        total = x.shape[0]
        x_fp16 = x.half()
        if total <= self.batch_size:
            self.d_input[:total].copy_(x_fp16)
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            return self.d_output[:total].float()
        results = []
        for s in range(0, total, self.batch_size):
            e = min(s + self.batch_size, total)
            bs = e - s
            self.d_input[:bs].copy_(x_fp16[s:e])
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            results.append(self.d_output[:bs].float().clone())
        return torch.cat(results, dim=0)


def build_engine_once():
    """主进程构建 FP16 engine (一次)"""
    import tensorrt as trt
    if os.path.exists(ENGINE_PATH):
        print(f"复用已有 engine: {ENGINE_PATH}")
        return
    os.makedirs(os.path.dirname(ENGINE_PATH), exist_ok=True)
    from models import get_model
    from general_utils.config_utils import load_config
    torch.cuda.set_device(0)
    cfg = load_config(os.path.join(CKPT_PATH, 'model.yaml'))
    cfg.start_from = ''
    cfg.freeze = False
    model = get_model(cfg, 'work_0605')
    model.load_state_dict_from_path(os.path.join(CKPT_PATH, 'model.pt'))
    model = model.half().cuda().eval()
    onnx_path = ENGINE_PATH.replace('.engine', '.onnx')
    dummy = torch.randn(BATCH_SIZE, 3, 112, 112, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(model, dummy, onnx_path, input_names=['input'],
                          output_names=['output'], opset_version=17, dynamo=False)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        assert parser.parse(f.read()), "ONNX 解析失败"
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    serialized = builder.build_serialized_network(network, config)
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized)
    os.remove(onnx_path)
    del model
    torch.cuda.empty_cache()
    print(f"engine 构建完成: {ENGINE_PATH}")


def _build_dataset():
    from torchvision.datasets import ImageFolder
    return ImageFolder(DATA_PATH, transform=get_transform())


# ============ 真实数据评估 worker ============
def worker_eval(rank, world_size, t_dispatch, ret_dict):
    """端到端: 启动 -> engine init -> DataLoader 推理. 各阶段计时写入 ret_dict[rank]."""
    t_enter = time.time()
    torch.cuda.set_device(rank)

    dataset = _build_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, collate_fn=_collate,
                            pin_memory=True, persistent_workers=True)

    t_eng0 = time.time()
    infer = TRTInfer(ENGINE_PATH, batch_size=BATCH_SIZE)
    # warmup 1 batch (engine 首次执行有 lazy 开销)
    _w = torch.zeros(BATCH_SIZE, 3, 112, 112, device='cuda')
    infer(_w)
    torch.cuda.synchronize()
    t_eng1 = time.time()

    n_img = 0
    t_inf0 = time.time()
    for batch in dataloader:
        x = batch["pixel_values"].cuda(non_blocking=True)
        x_flip = torch.flip(x, dims=[3])
        x_combined = torch.cat([x, x_flip], dim=0)
        with torch.no_grad():
            feats = infer(x_combined)
        n_img += x.shape[0]
    torch.cuda.synchronize()
    t_inf1 = time.time()

    ret_dict[rank] = {
        'startup': t_enter - t_dispatch,        # 启动延迟 (spawn+import or 线程入口)
        'engine_init': t_eng1 - t_eng0,         # engine 反序列化+context+warmup
        'infer': t_inf1 - t_inf0,               # 数据循环推理
        'n_img': n_img,
        't_enter': t_enter, 't_done': t_inf1,
    }


# ============ 纯推理 micro-bench worker (无数据 IO) ============
def worker_microbench(rank, world_size, t_dispatch, ret_dict, n_iter=200):
    t_enter = time.time()
    torch.cuda.set_device(rank)
    t_eng0 = time.time()
    infer = TRTInfer(ENGINE_PATH, batch_size=BATCH_SIZE)
    x = torch.randn(BATCH_SIZE, 3, 112, 112, device='cuda')
    for _ in range(5):  # warmup
        infer(x)
    torch.cuda.synchronize()
    t_eng1 = time.time()

    t0 = time.time()
    for _ in range(n_iter):
        infer(x)
    torch.cuda.synchronize()
    t1 = time.time()
    ret_dict[rank] = {
        'startup': t_enter - t_dispatch,
        'engine_init': t_eng1 - t_eng0,
        'infer': t1 - t0,
        'n_img': n_iter * BATCH_SIZE,
        't_enter': t_enter, 't_done': t1,
    }


# ============ 调度: 多进程 ============
def run_multiprocess(world_size, worker_fn, **kw):
    manager = mp.Manager()
    ret_dict = manager.dict()
    t_dispatch = time.time()
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker_fn, args=(rank, world_size, t_dispatch, ret_dict), kwargs=kw)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    wall = time.time() - t_dispatch
    return dict(ret_dict), wall


# ============ 调度: 多线程 ============
def run_multithread(world_size, worker_fn, **kw):
    ret_dict = {}
    t_dispatch = time.time()
    threads = []
    for rank in range(world_size):
        t = threading.Thread(target=worker_fn, args=(rank, world_size, t_dispatch, ret_dict), kwargs=kw)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t_dispatch
    return ret_dict, wall


def summarize(name, ret, wall):
    ranks = sorted(ret.keys())
    startup = [ret[r]['startup'] for r in ranks]
    eng = [ret[r]['engine_init'] for r in ranks]
    inf = [ret[r]['infer'] for r in ranks]
    total_img = sum(ret[r]['n_img'] for r in ranks)
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    print(f"  启动延迟   : max={max(startup):.3f}s  mean={np.mean(startup):.3f}s")
    print(f"  engine init: max={max(eng):.3f}s  mean={np.mean(eng):.3f}s")
    print(f"  推理        : max={max(inf):.3f}s  mean={np.mean(inf):.3f}s")
    print(f"  端到端 wall : {wall:.3f}s")
    print(f"  总图片     : {total_img}  ->  吞吐 {total_img/wall:.0f} img/s (按wall)")
    return {'name': name, 'startup_max': max(startup), 'eng_max': max(eng),
            'inf_max': max(inf), 'wall': wall, 'total_img': total_img,
            'thr': total_img/wall}


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--mode', choices=['eval', 'micro', 'both'], default='both')
    args = parser.parse_args()

    print(f"构建 engine (一次性)...")
    build_engine_once()

    ng = args.num_gpu
    rows = []

    if args.mode in ('micro', 'both'):
        print(f"\n########## 纯推理 micro-bench (无数据IO, {ng}卡) ##########")
        ret, wall = run_multiprocess(ng, worker_microbench)
        rows.append(summarize(f'[MICRO] 多进程 x{ng}', ret, wall))
        gc.collect(); torch.cuda.empty_cache()
        ret, wall = run_multithread(ng, worker_microbench)
        rows.append(summarize(f'[MICRO] 多线程 x{ng}', ret, wall))
        gc.collect(); torch.cuda.empty_cache()

    if args.mode in ('eval', 'both'):
        print(f"\n########## 真实数据评估 ({DATA_PATH}, {ng}卡) ##########")
        ret, wall = run_multiprocess(ng, worker_eval)
        rows.append(summarize(f'[EVAL] 多进程 x{ng}', ret, wall))
        gc.collect(); torch.cuda.empty_cache()
        ret, wall = run_multithread(ng, worker_eval)
        rows.append(summarize(f'[EVAL] 多线程 x{ng}', ret, wall))

    print(f"\n\n{'#'*72}\n汇总\n{'#'*72}")
    print(f"{'方案':<22}{'启动(s)':>9}{'engInit(s)':>12}{'推理(s)':>10}{'wall(s)':>9}{'img/s':>9}")
    print('-'*72)
    for r in rows:
        print(f"{r['name']:<22}{r['startup_max']:>9.3f}{r['eng_max']:>12.3f}"
              f"{r['inf_max']:>10.3f}{r['wall']:>9.3f}{r['thr']:>9.0f}")

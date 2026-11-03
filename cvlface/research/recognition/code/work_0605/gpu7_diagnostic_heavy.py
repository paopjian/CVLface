"""
GPU 高强度压力测试 + 硬件故障检测 - 针对 RTX 4090 优化
目标：功率400W+，显存利用率60-70%，GPU利用率90%+

在纯压力之外做三件事（开销实测 <1%: 94.2 vs 94.4 steps/min）：
  1. 计算单元全覆盖: FP16/BF16/TF32/FP32/FP64/INT8/FP8 GEMM + Conv + SFU + 带宽
  2. 数值正确性校验: 固定输入 canary，检测静默计算错误 / bit-flip / NaN
  3. 硬件状态监控: 独立 nvidia-smi 进程采样功耗/频率/温度/降频原因/ECC + Xid 检查

================================ 用法 ================================

  # 快速筛查: 只跑计算单元扫描, 约15秒。多卡时能直接看出某卡某个单元是否异常
  python gpu7_diagnostic_heavy.py --gpu 7 --duration 0

  # 单卡完整压测 (单元扫描 + 30分钟持续压力 + 全程校验)
  python gpu7_diagnostic_heavy.py --gpu 7 --duration 30

  # 全卡并行压测 + 单元吞吐横向对比 (推荐: 坏卡在对比里最容易暴露)
  python gpu7_diagnostic_heavy.py --all --duration 30

  # 跳过单元扫描, 直接进持续压力
  python gpu7_diagnostic_heavy.py --gpu 7 --duration 30 --skip-sweep

  # 只压部分卡 (会自动用 UUID 对齐到 nvidia-smi 物理编号, 监控不会采错卡)
  CUDA_VISIBLE_DEVICES=3,5 python gpu7_diagnostic_heavy.py --all --duration 30

参数:
  --gpu N        单卡测试的 GPU 编号 (torch 索引), 默认 0
  --duration N   持续压力分钟数, 默认 30; 0 = 只跑单元扫描
  --all          测试所有可见 GPU (并行), 忽略 --gpu
  --skip-sweep   跳过计算单元扫描

============================ 输出怎么看 ============================

单元扫描 (RTX 4090 实测参考值, 偏低 20% 以上才需怀疑):
  FP16/BF16 ~160/148 TF | TF32 ~80 TF | FP32 ~56 TF (CUDA core) | FP64 ~1.2 TF
  INT8 ~171 TOPS | FP8 ~335 TF | Conv FP16 ~91 TF | 带宽 ~870 GB/s
  行首 '!' = 该单元数值校验未通过; '-' = 该单元不支持 (非故障)
  注: Conv2d 喂 FP32 也走 TF32 tensor core (cudnn.allow_tf32 默认 True), 不是 FP32 单元

数值校验 (两级判据, 缺一不可):
  硬错误  相对误差 > 基线x20。对 FP32 灵敏 (抓 1e-2 级扰动), 对低精度钝
  告警    与首轮结果非 bit-wise 一致。同卡同 shape 下 cuBLAS kernel 选择稳定,
          结果应逐位可复现; 这一档兜住低精度的漏检 (BF16 硬判据连 +1.0 都漏)
  故障注入实测: 20 轮零误报, FP16/BF16/FP32 微扰动 + VRAM bit-flip 全部检出

硬件状态 (降频按"占多少比例采样点"报, 不是整场 OR 一次):
  SwPowerCap        满载必现, 不是问题
  SwThermal/HwThermal/HwSlowdown  温度到限降频 -> 散热问题, 不是显卡坏
                    <5% 时间只提示, >=5% 进告警 (持续热降频才实际损失吞吐)
  HwPowerBrake      真硬件异常 (外部供电拉闸)
  4090 温度线 (nvidia-smi -q -d TEMPERATURE): Target 84C 起热降频,
    约 86C 硬件降频, 约 91C 保护关机。注意 T.Limit 报的是"距限温差多少度"的
    相对值, 不是绝对温度
  告警门槛: 功耗<400W / util<90% / 平均温度>=84C / 峰值>=86C / 峰值>=90C / ECC非零
  ECC 在 GeForce 上恒为 N/A, 按不支持处理
  dmesg 需 root 才读得到; 读不到会提示手动 dmesg | grep -i xid

结论行 ✅ 表示无计算错误且无数值偏差; ⚠️ 告警不等于故障, 看具体项
"""
import os
import re
import time
import tempfile
import threading
import subprocess
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

# nvidia-smi clocks_throttle_reasons.active 位含义, 第三列是性质分类:
#   'ok'      满载/空闲时的正常现象, 不告警
#   'thermal' 温度到限导致的降频 -> 散热问题, 不是显卡坏
#   'hw'      真正的硬件级异常 (外部供电拉闸等)
# RTX 4090 温度参考 (nvidia-smi -q -d TEMPERATURE): Target 84°C,
#   Slowdown 约 86°C, Shutdown 约 91°C
THROTTLE_FLAGS = [
    (0x0001, 'GpuIdle',         'ok'),
    (0x0002, 'AppClockSetting', 'ok'),
    (0x0004, 'SwPowerCap',      'ok'),
    (0x0008, 'HwSlowdown',      'thermal'),   # 多与 HwThermal 同时出现
    (0x0010, 'SyncBoost',       'ok'),
    (0x0020, 'SwThermal',       'thermal'),
    (0x0040, 'HwThermal',       'thermal'),
    (0x0080, 'HwPowerBrake',    'hw'),
    (0x0100, 'DisplayClock',    'ok'),
]
TEMP_TARGET = 84      # 到此开始热降频
TEMP_SLOWDOWN = 86    # 硬件降频
TEMP_SHUTDOWN = 91    # 保护性关机


def decode_throttle(mask):
    """位掩码 -> (原因名列表, 是否含硬件级异常)"""
    names, bad = [], False
    for bit, name, kind in THROTTLE_FLAGS:
        if mask & bit:
            names.append(name)
            bad = bad or (kind == 'hw')
    return names, bad


def phys_index_map():
    """torch 索引 -> nvidia-smi 物理索引。
    CUDA_VISIBLE_DEVICES 存在时两者不一致, 靠 UUID 对齐, 否则监控会采错卡。
    """
    fallback = {i: i for i in range(torch.cuda.device_count())}
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10).stdout
        uuid2phys = {}
        for line in out.strip().splitlines():
            idx, uu = [x.strip() for x in line.split(',')]
            uuid2phys[uu.replace('GPU-', '')] = int(idx)
    except Exception:
        return fallback
    m = {}
    for i in range(torch.cuda.device_count()):
        u = str(getattr(torch.cuda.get_device_properties(i), 'uuid', '')).replace('GPU-', '')
        m[i] = uuid2phys.get(u, i)
    return m

class GPUMonitor:
    """独立 nvidia-smi 子进程采样硬件状态。
    不用 Python 线程: 压力测试会长时间持有 GIL, 采样线程会被饿死采不到数。
    """

    def __init__(self, phys_ids, interval_ms=500):
        self.phys_ids = list(phys_ids)
        self.proc = None
        self.logfile = None
        self.interval_ms = interval_ms

    def start(self):
        self.logfile = tempfile.NamedTemporaryFile('w+', suffix='.csv', prefix='gpumon_', delete=False)
        q = ('index,power.draw,clocks.sm,temperature.gpu,utilization.gpu,'
             'clocks_throttle_reasons.active,ecc.errors.uncorrected.volatile.total')
        try:
            self.proc = subprocess.Popen(
                ['nvidia-smi', '-i', ','.join(str(i) for i in self.phys_ids),
                 f'--query-gpu={q}', '--format=csv,noheader,nounits',
                 '-lms', str(self.interval_ms)],
                stdout=self.logfile, stderr=subprocess.DEVNULL)
        except Exception:
            self.proc = None

    def stop(self):
        if self.proc:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.logfile:
            self.logfile.flush()

    def summary(self):
        """-> {phys_id: dict}, 丢弃前 20% 升温样本"""
        if not self.logfile:
            return {}
        rows = {}
        try:
            with open(self.logfile.name) as f:
                for line in f:
                    p = [x.strip() for x in line.split(',')]
                    if len(p) < 7:
                        continue
                    try:
                        gid = int(p[0])
                        vals = (float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5], 16))
                    except ValueError:
                        continue
                    ecc = None
                    if p[6] not in ('[N/A]', 'N/A', '[Not Supported]'):
                        try:
                            ecc = int(p[6])
                        except ValueError:
                            pass
                    rows.setdefault(gid, []).append(vals + (ecc,))
        except Exception:
            return {}

        out = {}
        for gid, v in rows.items():
            v = v[len(v) // 5:] or v
            pw = [x[0] for x in v]
            ck = [x[1] for x in v]
            tp = [x[2] for x in v]
            ut = [x[3] for x in v]
            # 按位统计出现比例, 而不是整场 OR。瞬时降频和持续降频是两回事:
            # 一次 HwThermal 采样和 20 分钟 HwThermal 都会 OR 出同一个掩码。
            ratio = {}
            for bit, name, _ in THROTTLE_FLAGS:
                c = sum(1 for x in v if x[4] & bit)
                if c:
                    ratio[name] = c / len(v)
            eccs = [x[5] for x in v if x[5] is not None]
            out[gid] = dict(
                n=len(v), pw_avg=sum(pw) / len(pw), pw_max=max(pw), pw_min=min(pw),
                ck_avg=sum(ck) / len(ck), ck_min=min(ck),
                tp_avg=sum(tp) / len(tp), tp_max=max(tp),
                ut_avg=sum(ut) / len(ut),
                throttle=ratio,
                ecc=max(eccs) if eccs else None,
            )
        return out

    def cleanup(self):
        try:
            if self.logfile:
                os.unlink(self.logfile.name)
        except Exception:
            pass


def check_xid(since_ts=None):
    """dmesg 里找 Xid。返回 (行列表, 是否可读)"""
    try:
        out = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return [], False
        lines = [l for l in out.stdout.splitlines() if re.search(r'Xid', l)]
        return lines[-10:], True
    except Exception:
        return [], False

class Canary:
    """数值正确性校验。
    固定输入 + CPU float64 参考值。GPU 只要算错一次就能抓到, 覆盖压力测试
    抓不到的静默错误 (tensor core 算错 / 显存 bit-flip / 计算结果漂移)。
    参考值必须在 CPU 上算: 用 GPU 自己算参考, 坏卡会自证清白。
    """

    N = 1024

    def __init__(self, device):
        self.device = device
        g = torch.Generator().manual_seed(0x5EED)
        a = torch.randn(self.N, self.N, generator=g)
        b = torch.randn(self.N, self.N, generator=g)
        ref = (a.double() @ b.double())
        self.ref_norm = ref.abs().max().item()
        self.ref = ref.to(device)
        self.a16 = a.half().to(device)
        self.b16 = b.half().to(device)
        self.abf = a.bfloat16().to(device)
        self.bbf = b.bfloat16().to(device)
        self.a32 = a.to(device)
        self.b32 = b.to(device)
        # 基线相对误差 = 该精度的固有舍入水平, 后续以它的 20 倍为告警阈值
        self.base = {}
        self.first = {}
        for k in ('fp16', 'bf16', 'fp32'):
            r = self._once(k)
            self.base[k] = self._relerr(k, r)
            self.first[k] = r.clone()
        # 大张量求和: 覆盖 ballast 之外的显存读取通路和归约树
        gg = torch.Generator().manual_seed(0xB1A5)
        self.red = torch.randn(1 << 22, generator=gg).to(device)
        self.red_ref = self.red.double().sum().item()

        self.fails = []
        self.bit_warns = []
        self.checks = 0

    def _once(self, kind):
        if kind == 'fp16':
            return torch.mm(self.a16, self.b16)
        if kind == 'bf16':
            return torch.mm(self.abf, self.bbf)
        return torch.mm(self.a32, self.b32)

    def _relerr(self, kind, r):
        return ((r.double() - self.ref).abs().max() / self.ref_norm).item()

    def check(self, elapsed_min):
        """跑一轮校验, 返回 True=全部通过。约 2ms。

        两级判据, 因为单一判据都不够:
          1) 相对误差 > 基线x20 -> 硬错误。对 fp32 灵敏 (能抓 1e-2 级扰动),
             但对低精度很钝 (bf16 基线本身 3.7e-3, 单元素扰动埋在噪声里)。
          2) 与首轮结果非 bit-wise 一致 -> 告警。同卡同 shape 下 cuBLAS
             kernel 选择稳定, 结果应逐位可复现, 这一档能补上低精度的漏检。
        """
        self.checks += 1
        ok = True
        for k in ('fp16', 'bf16', 'fp32'):
            try:
                r = self._once(k)
                err = self._relerr(k, r)
                thr = max(self.base[k] * 20, 1e-9)
                if err > thr:
                    self.fails.append((elapsed_min, f'{k} 数值错误',
                                       f'rel_err={err:.3e} > 阈值{thr:.3e} (基线{self.base[k]:.3e})'))
                    ok = False
                elif not torch.equal(r, self.first[k]):
                    d = (r.double() - self.first[k].double()).abs().max().item()
                    n = int((r != self.first[k]).sum().item())
                    self.bit_warns.append((elapsed_min, k, d, n))
            except Exception as e:
                self.fails.append((elapsed_min, f'{k} 校验异常', str(e)))
                ok = False
        try:
            s = self.red.double().sum().item()
            if self.red_ref != 0 and abs(s - self.red_ref) / abs(self.red_ref) > 1e-12:
                self.fails.append((elapsed_min, '大张量归约错误',
                                   f'sum={s:.10e} 期望 {self.red_ref:.10e}'))
                ok = False
        except Exception as e:
            self.fails.append((elapsed_min, '归约校验异常', str(e)))
            ok = False
        return ok


class Ballast:
    """显存填充块。顺便当 VRAM 校验用: 每块填独立数值, 定期查 min/max。
    能抓 bit-flip 和地址译码错误, 读一遍 ~10GB 只要十几毫秒。
    """

    def __init__(self, device, blocks, block_elems=512 * 512):
        self.device = device
        self.blocks = []
        self.vals = []
        for i in range(blocks):
            try:
                t = torch.empty(block_elems, device=device, dtype=torch.float16)
            except torch.cuda.OutOfMemoryError:
                break
            v = float((i % 61) - 30)          # -30..30, fp16 精确可表示
            t.fill_(v)
            self.blocks.append(t)
            self.vals.append(v)
        self.errors = []

    @property
    def gb(self):
        if not self.blocks:
            return 0.0
        return len(self.blocks) * self.blocks[0].numel() * 2 / 1024 ** 3

    def verify(self, elapsed_min):
        bad = 0
        for t, v in zip(self.blocks, self.vals):
            if t.min().item() != v or t.max().item() != v:
                bad += 1
        if bad:
            self.errors.append((elapsed_min, 'VRAM 校验', f'{bad}/{len(self.blocks)} 块数据损坏'))
            return False
        return True

SWEEP_LOCK = threading.Lock()   # 多卡时串行做基准测量, 见 _bench


def _bench(fn, device, warmup=3, iters=6, reps=3):
    """测单个算子的 GPU 侧耗时, 取 reps 次里最快的一次。

    三个坑都必须避开, 否则多卡并行时数字完全失真 (实测出现过 18000 TFLOPS):
      1) torch.cuda.synchronize() 不带参数只同步"调用线程的当前设备"(默认 0)。
         别的线程测别的卡时会立刻返回, 根本没等自己的 kernel 跑完。
      2) time.time() 测的是墙钟。Python 侧启 kernel 要抢 GIL, 7 个线程互相抢时
         调度延迟会被算进耗时。改用 CUDA event 测 GPU 侧时间。
      3) 即便用 event, 若 GIL 卡住导致 kernel 队列排空, 空隙仍计入 event 区间。
         所以取多次中的最小耗时: 峰值吞吐是最少被干扰污染的估计。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    best = float('inf')
    for _ in range(reps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            fn()
        ev1.record()
        torch.cuda.synchronize(device)
        best = min(best, ev0.elapsed_time(ev1) / 1e3 / iters)
    return best


def unit_sweep(gpu_id, silent=False, serialize=False):
    """遍历所有计算单元, 同时测吞吐和正确性。约 10-15 秒。
    RTX 4090 参考值 (实测): FP16/BF16 ~160/148, TF32 ~80, FP32 ~56,
                            FP64 ~1.2, INT8 ~171, FP8 ~335 TFLOPS

    serialize=True: 多卡模式下用全局锁串行测量。基准测量必须独占, 否则 7 张卡
    同时启 kernel 抢 GIL, 结果失真到不能用 (同一卡两轮能差 10 倍)。代价是
    7 卡多花约 100 秒, 相对 30 分钟压测可忽略。
    返回 (结果列表, 失败列表)
    """
    if serialize:
        with SWEEP_LOCK:
            return _unit_sweep_impl(gpu_id, silent)
    return _unit_sweep_impl(gpu_id, silent)


def _unit_sweep_impl(gpu_id, silent=False):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)       # synchronize(device) 之外, cuBLAS handle 也按当前设备取
    N, M = 4096, 512
    res, fails = [], []

    g = torch.Generator().manual_seed(1234)
    sa = torch.randn(M, M, generator=g)
    sb = torch.randn(M, M, generator=g)
    sref = (sa.double() @ sb.double())
    snorm = sref.abs().max().item()
    sref = sref.to(device)

    def relerr(r):
        return ((r.double() - sref).abs().max() / snorm).item()

    def entry(name, big_fn, flops, small_fn, tol, note=''):
        try:
            t = _bench(big_fn, device)
            tflops = flops / t / 1e12
            err = relerr(small_fn()) if small_fn else float('nan')
            bad = (small_fn is not None) and (err > tol)
            if bad:
                fails.append((0.0, f'{name} 单元数值错误', f'rel_err={err:.2e} > {tol:.0e}'))
            res.append((name, tflops, err, note, not bad))
        except Exception as e:
            # 不支持的单元 (如老卡没有 FP8) 记为 N/A, 不算故障
            res.append((name, float('nan'), float('nan'), f'{type(e).__name__}', None))

    gf = 2 * N ** 3
    a16 = torch.randn(N, N, device=device, dtype=torch.float16)
    b16 = torch.randn(N, N, device=device, dtype=torch.float16)
    c16 = torch.empty(N, N, device=device, dtype=torch.float16)
    sa16, sb16 = sa.half().to(device), sb.half().to(device)
    entry('FP16 GEMM (tensor core)', lambda: torch.mm(a16, b16, out=c16), gf,
          lambda: torch.mm(sa16, sb16), 2e-2)

    abf, bbf = a16.bfloat16(), b16.bfloat16()
    cbf = torch.empty(N, N, device=device, dtype=torch.bfloat16)
    sabf, sbbf = sa.bfloat16().to(device), sb.bfloat16().to(device)
    entry('BF16 GEMM (tensor core)', lambda: torch.mm(abf, bbf, out=cbf), gf,
          lambda: torch.mm(sabf, sbbf), 1e-1, '训练用 bf16-mixed')
    del abf, bbf, cbf

    a32 = torch.randn(N, N, device=device)
    b32 = torch.randn(N, N, device=device)
    c32 = torch.empty(N, N, device=device)
    sa32, sb32 = sa.to(device), sb.to(device)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    entry('FP32 GEMM (CUDA core)', lambda: torch.mm(a32, b32, out=c32), gf,
          lambda: torch.mm(sa32, sb32), 1e-5)
    torch.backends.cuda.matmul.allow_tf32 = True
    entry('TF32 GEMM (tensor core)', lambda: torch.mm(a32, b32, out=c32), gf,
          lambda: torch.mm(sa32, sb32), 1e-2)
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    del a32, b32, c32

    a64 = torch.randn(2048, 2048, device=device, dtype=torch.float64)
    b64 = torch.randn(2048, 2048, device=device, dtype=torch.float64)
    entry('FP64 GEMM (1/64 速率)', lambda: torch.mm(a64, b64), 2 * 2048 ** 3,
          lambda: torch.mm(sa.double().to(device), sb.double().to(device)), 1e-12)
    del a64, b64

    try:
        ai = torch.randint(-8, 8, (N, N), device=device, dtype=torch.int8)
        bi = torch.randint(-8, 8, (N, N), device=device, dtype=torch.int8)
        si_a = torch.randint(-8, 8, (M, M), dtype=torch.int8)
        si_b = torch.randint(-8, 8, (M, M), dtype=torch.int8)
        iref = (si_a.double() @ si_b.double()).to(device)
        t = _bench(lambda: torch._int_mm(ai, bi), device)
        got = torch._int_mm(si_a.to(device), si_b.to(device))
        d = (got.double() - iref).abs().max().item()
        if d != 0:
            fails.append((0.0, 'INT8 单元数值错误', f'整数运算出现偏差 {d}'))
        res.append(('INT8 GEMM (tensor core)', gf / t / 1e12, d, '整数精确校验', d == 0))
        del ai, bi
    except Exception as e:
        res.append(('INT8 GEMM (tensor core)', float('nan'), float('nan'), f'{type(e).__name__}', None))

    try:
        af8 = a16.to(torch.float8_e4m3fn)
        bf8 = b16.t().contiguous().to(torch.float8_e4m3fn)
        sc = torch.tensor(1.0, device=device)
        entry('FP8 e4m3 GEMM (sm89)',
              lambda: torch._scaled_mm(af8, bf8.t(), scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16),
              gf, None, 0.0, '仅测吞吐')
        del af8, bf8
    except Exception as e:
        res.append(('FP8 e4m3 GEMM (sm89)', float('nan'), float('nan'), f'{type(e).__name__}', None))
    del a16, b16, c16

    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    conv = nn.Conv2d(128, 256, 3, padding=1).to(device)
    cx = torch.randn(32, 128, 224, 224, device=device)
    cflops = 2 * 32 * 256 * 224 * 224 * 128 * 9
    try:
        t = _bench(lambda: conv(cx), device, 3, 5)
        res.append(('Conv2d FP32 输入 (cuDNN)', cflops / t / 1e12, float('nan'),
                    'cudnn.allow_tf32 默认 True -> 实走 TF32', True))
        ch, xh = conv.half(), cx.half()
        t = _bench(lambda: ch(xh), device, 3, 5)
        res.append(('Conv2d FP16 (tensor core)', cflops / t / 1e12, float('nan'), '', True))
    except Exception as e:
        res.append(('Conv2d', float('nan'), float('nan'), f'{type(e).__name__}', None))
    torch.backends.cudnn.benchmark = prev_bench
    del conv, cx

    torch.cuda.empty_cache()
    extra = []
    try:
        v = torch.randn(1 << 24, device=device)
        t = _bench(lambda: (torch.exp(v), torch.sin(v), torch.rsqrt(v)), device)
        extra.append(('SFU exp/sin/rsqrt', f'{3*v.numel()/t/1e9:.1f} Gelem/s'))
        big = torch.empty(1 << 27, device=device, dtype=torch.float32)   # 512MB
        dst = torch.empty_like(big)
        t = _bench(lambda: dst.copy_(big), device)
        extra.append(('显存带宽 D2D', f'{2*big.numel()*4/t/1e9:.0f} GB/s (理论1008)'))
        del big, dst, v
        h = torch.empty(1 << 24, dtype=torch.float32, pin_memory=True)   # 64MB
        d = torch.empty(1 << 24, device=device)
        t = _bench(lambda: d.copy_(h), device, 2, 5)
        extra.append(('PCIe H2D', f'{h.numel()*4/t/1e9:.1f} GB/s'))
        t = _bench(lambda: h.copy_(d), device, 2, 5)
        extra.append(('PCIe D2H', f'{h.numel()*4/t/1e9:.1f} GB/s'))
        del h, d
    except Exception as e:
        extra.append(('带宽测试', f'失败: {type(e).__name__}'))
    torch.cuda.empty_cache()

    if not silent:
        print(f"  {'单元':<28}{'吞吐':>12}   {'相对误差':>10}  说明")
        print(f"  {'-'*72}")
        for name, tf, err, note, ok in res:
            mark = ' ' if ok else ('!' if ok is False else '-')
            tf_s = f'{tf:8.1f} TF' if tf == tf else '     N/A'
            err_s = f'{err:9.2e}' if err == err else '        -'
            print(f" {mark}{name:<28}{tf_s:>12}   {err_s:>10}  {note}")
        for name, val in extra:
            print(f"  {name:<28}{val:>12}")
    return res, fails


class ExtraUnits:
    """主循环里轮转的额外单元 (FP8/INT8/FP64/SFU)。
    单次 burst 约 5ms, 每 15 秒调一次 -> 开销可忽略, 但这些单元不会整场闲置。
    """

    def __init__(self, device):
        self.ok = True
        n = 2048
        try:
            self.a64 = torch.randn(1024, 1024, device=device, dtype=torch.float64)
            self.b64 = torch.randn(1024, 1024, device=device, dtype=torch.float64)
            self.ai = torch.randint(-8, 8, (n, n), device=device, dtype=torch.int8)
            self.bi = torch.randint(-8, 8, (n, n), device=device, dtype=torch.int8)
            self.v = torch.randn(1 << 22, device=device)
            h = torch.randn(n, n, device=device, dtype=torch.float16)
            self.af8 = h.to(torch.float8_e4m3fn)
            self.bf8 = h.t().contiguous().to(torch.float8_e4m3fn)
            self.sc = torch.tensor(1.0, device=device)
            del h
        except Exception:
            self.ok = False

    def burst(self):
        if not self.ok:
            return
        torch.mm(self.a64, self.b64)
        torch._int_mm(self.ai, self.bi)
        torch.exp(self.v)
        torch.rsqrt(self.v.abs() + 1)
        try:
            torch._scaled_mm(self.af8, self.bf8.t(), scale_a=self.sc, scale_b=self.sc,
                             out_dtype=torch.bfloat16)
        except Exception:
            pass

CHECK_INTERVAL_SEC = 15.0   # 数值校验 / VRAM 校验 / 额外单元 burst 间隔


class HeavyModel(nn.Module):
    """8层 x 8192 宽, ~436M 参数"""

    def __init__(self):
        super().__init__()
        L = [nn.Linear(2048, 8192), nn.BatchNorm1d(8192), nn.GELU()]
        for _ in range(6):
            L += [nn.Linear(8192, 8192), nn.BatchNorm1d(8192), nn.GELU()]
        L += [nn.Linear(8192, 2048)]
        self.layers = nn.Sequential(*L)

    def forward(self, x):
        return self.layers(x)


def stress_test_heavy(gpu_id, duration_min=30, silent=False, skip_sweep=False):
    """单卡高强度压力测试 + 数值校验 + 单元扫描

    Args:
        gpu_id: GPU 编号 (torch 索引)
        duration_min: 持续时间(分钟), <=0 表示只跑单元扫描
        silent: 静默模式, 多卡并行时由主线程汇总输出
        skip_sweep: 跳过单元扫描
    Returns:
        dict(steps, minutes, throughput, errors, bit_warns, checks, sweep, mem_pct)
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(gpu_id)
    total_mem = props.total_memory

    out = dict(steps=0, minutes=0.0, throughput=0.0, errors=[], bit_warns=[],
               checks=0, sweep=[], mem_pct=0.0)

    if not silent:
        print(f"\n{'='*74}")
        print(f"  GPU {gpu_id}: {props.name}")
        print(f"  显存: {total_mem/1024**3:.1f}GB | SM: {props.multi_processor_count} | sm_{props.major}{props.minor}")
        print(f"  目标: 功率>400W, GPU利用率>90%, 显存利用率60-70%")
        print(f"{'='*74}\n")

    # ---------- 阶段1: 计算单元扫描 ----------
    if not skip_sweep:
        if not silent:
            print("  [阶段1] 计算单元扫描 (吞吐 + 数值正确性, 约15秒)\n")
        try:
            # silent=True 即多卡并行, 此时必须串行测量
            sweep, fails = unit_sweep(gpu_id, silent=silent, serialize=silent)
            out['sweep'] = sweep
            out['errors'].extend(fails)
        except Exception as e:
            out['errors'].append((0.0, '单元扫描', str(e)))
        if not silent:
            print()

    if duration_min <= 0:
        return out

    # ---------- 阶段2: 持续压力 ----------
    if not silent:
        print("  [阶段2] 持续压力测试\n")
        print("  初始化大模型...")
    try:
        model = HeavyModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        n_params = sum(p.numel() for p in model.parameters())
        if not silent:
            print(f"    模型参数: {n_params/1e6:.1f}M")

        batch_size = 2048
        x_buf = torch.randn(batch_size, 2048, device=device)
        y_buf = torch.randn(batch_size, 2048, device=device)

        # FP16/BF16 共用同一块存储: 都是 2 字节, view 过去不额外占显存
        mat_size = 12288
        if not silent:
            print(f"  分配 FP16/BF16 矩阵: {mat_size}x{mat_size}...")
        mat_a_f16 = torch.randn(mat_size, mat_size, device=device, dtype=torch.float16)
        mat_b_f16 = torch.randn(mat_size, mat_size, device=device, dtype=torch.float16)
        mat_c_f16 = torch.empty(mat_size, mat_size, device=device, dtype=torch.float16)
        mat_a_bf16 = mat_a_f16.view(torch.bfloat16)
        mat_b_bf16 = mat_b_f16.view(torch.bfloat16)
        mat_c_bf16 = mat_c_f16.view(torch.bfloat16)

        if not silent:
            print(f"  分配 FP32 矩阵: {mat_size//2}x{mat_size//2}...")
        mat_a_f32 = torch.randn(mat_size//2, mat_size//2, device=device, dtype=torch.float32)
        mat_b_f32 = torch.randn(mat_size//2, mat_size//2, device=device, dtype=torch.float32)
        mat_c_f32 = torch.empty(mat_size//2, mat_size//2, device=device, dtype=torch.float32)

        if not silent:
            print(f"  分配卷积层...")
        conv = nn.Conv2d(128, 256, 3, padding=1).to(device)
        conv_buf = torch.randn(32, 128, 224, 224, device=device)

        if not silent:
            print("  初始化数值校验 canary (CPU float64 参考)...")
        canary = Canary(device)
        extra = ExtraUnits(device)
    except Exception as e:
        out['errors'].append((0.0, '初始化', str(e)))
        if not silent:
            print(f"\n  初始化失败: {e}")
        return out

    def one_step(step_idx):
        """一个压力步。fp16/bf16 交替, 让两种 tensor core 路径都跑到。"""
        model.train()
        x_buf.normal_()
        y_buf.normal_()
        amp_dtype = torch.bfloat16 if step_idx % 2 else torch.float16
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss = crit(model(x_buf), y_buf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        use_bf16 = bool(step_idx % 2)
        ma = mat_a_bf16 if use_bf16 else mat_a_f16
        mb = mat_b_bf16 if use_bf16 else mat_b_f16
        mc = mat_c_bf16 if use_bf16 else mat_c_f16
        with torch.cuda.stream(stream_1):
            for _ in range(10):
                ma.normal_()
                mb.normal_()
                torch.mm(ma, mb, out=mc)
        with torch.cuda.stream(stream_2):
            for _ in range(10):
                torch.mm(mc, ma, out=mb)
        with torch.cuda.stream(stream_fp32):
            for _ in range(6):
                mat_a_f32.normal_()
                mat_b_f32.normal_()
                torch.mm(mat_a_f32, mat_b_f32, out=mat_c_f32)
        with torch.cuda.stream(stream_conv):
            conv_buf.normal_()
            conv(conv_buf)
        return loss

    stream_1 = torch.cuda.Stream(device=device)
    stream_2 = torch.cuda.Stream(device=device)
    stream_fp32 = torch.cuda.Stream(device=device)
    stream_conv = torch.cuda.Stream(device=device)

    # 先跑几步让 AdamW 状态和激活值都分配出来, 再按实际峰值决定 ballast 大小。
    # 原来按启动时占用去凑 55% 会低估: optimizer state 是首次 step 后才分配的。
    try:
        for i in range(3):
            one_step(i)
        torch.cuda.synchronize(device)
    except Exception as e:
        out['errors'].append((0.0, '预热步', str(e)))
        return out

    # 按 nvidia-smi 视角 (含 CUDA context) 补到 ~68%, 而不是按 allocated 估
    free_b, total_b = torch.cuda.mem_get_info(device)
    used_b = total_b - free_b
    peak = torch.cuda.max_memory_reserved(device)
    block_bytes = 512 * 512 * 2                      # 512x512 fp16 = 512KB
    n_block = max(0, (int(total_b * 0.68) - used_b) // block_bytes)
    ballast = Ballast(device, int(n_block))
    free_b, _ = torch.cuda.mem_get_info(device)
    used_b = total_b - free_b
    out['mem_pct'] = used_b / total_b * 100
    if not silent:
        print(f"\n  配置完成:")
        print(f"    运行峰值 {peak/1024**3:.1f}GB(reserved) + ballast {ballast.gb:.1f}GB"
              f" -> 显存占用 {used_b/1024**3:.1f}/{total_b/1024**3:.1f}GB ({out['mem_pct']:.0f}%)")
        print(f"    批次大小: {batch_size}")
        print(f"    矩阵规模: FP16/BF16 {mat_size}x{mat_size}, FP32 {mat_size//2}x{mat_size//2}")
        print(f"    卷积输入: [32, 128, 224, 224]")
        print(f"    校验: 每 {CHECK_INTERVAL_SEC:.0f}s 做 canary + VRAM + FP8/INT8/FP64/SFU burst\n")

    start = time.time()
    dur_sec = duration_min * 60
    step = 0
    loss = None

    if not silent:
        print(f"  开始时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"  预计结束: {datetime.fromtimestamp(start+dur_sec).strftime('%H:%M:%S')}")
        print(f"  持续时间: {duration_min}分钟\n")

    pbar = tqdm(total=dur_sec, desc="  压力测试", unit="s", ncols=104) if not silent else None
    last_update = start
    last_check = start

    try:
        while time.time() - start < dur_sec:
            elapsed_min = (time.time() - start) / 60
            try:
                loss = one_step(step)
            except Exception as e:
                out['errors'].append((elapsed_min, '压力步', str(e)))
                break
            step += 1

            now = time.time()
            if now - last_check >= CHECK_INTERVAL_SEC:
                try:
                    torch.cuda.synchronize(device)
                    if not torch.isfinite(loss).all():
                        out['errors'].append((elapsed_min, 'loss 非有限值',
                                              f'loss={loss.item()} (NaN/Inf)'))
                    canary.check(elapsed_min)
                    ballast.verify(elapsed_min)
                    extra.burst()
                    torch.cuda.synchronize(device)
                except Exception as e:
                    out['errors'].append((elapsed_min, '周期校验', str(e)))
                    break
                last_check = now

            if now - last_update >= 2.0:
                torch.cuda.synchronize(device)
                if pbar:
                    pbar.n = min(now - start, dur_sec)
                    fb, tb = torch.cuda.mem_get_info(device)
                    mem_gb = (tb - fb) / 1024**3      # 与 nvidia-smi 一致
                    nerr = len(out['errors']) + len(canary.fails) + len(ballast.errors)
                    pbar.set_postfix_str(
                        f"step={step} loss={loss.item():.3f} mem={mem_gb:.1f}GB "
                        f"chk={canary.checks} err={nerr}")
                    pbar.refresh()
                last_update = now

    except KeyboardInterrupt:
        if not silent:
            print("\n  用户中断")

    if pbar:
        pbar.close()
    try:
        torch.cuda.synchronize(device)
    except Exception as e:
        out['errors'].append(((time.time()-start)/60, '最终同步', str(e)))

    out['errors'].extend(canary.fails)
    out['errors'].extend(ballast.errors)
    out['bit_warns'] = canary.bit_warns
    out['checks'] = canary.checks
    out['minutes'] = (time.time() - start) / 60
    out['steps'] = step
    out['throughput'] = step / out['minutes'] if out['minutes'] > 0 else 0.0

    del ballast, canary, extra
    del mat_a_f16, mat_b_f16, mat_c_f16, mat_a_f32, mat_b_f32, mat_c_f32
    del x_buf, y_buf, conv_buf, model, opt
    torch.cuda.empty_cache()
    return out


def print_hw(mon_summary, phys_id, indent='  ', check_load=True):
    """打印单卡硬件监控结果, 返回告警列表。
    check_load=False 时不判功耗/利用率 (只跑单元扫描的场景, 这两项本就不该满)。
    """
    warns = []
    s = mon_summary.get(phys_id)
    if not s:
        print(f"{indent}硬件监控: 无采样数据 (nvidia-smi 不可用?)")
        return warns
    thr = s['throttle']
    thr_s = ' '.join(f'{k}={v*100:.0f}%' for k, v in
                     sorted(thr.items(), key=lambda x: -x[1])) or 'None'
    print(f"{indent}功耗 {s['pw_avg']:.0f}W avg / {s['pw_max']:.0f}W peak / {s['pw_min']:.0f}W min")
    print(f"{indent}SM频率 {s['ck_avg']:.0f}MHz avg / {s['ck_min']:.0f}MHz min | "
          f"温度 {s['tp_avg']:.1f}°C avg / {s['tp_max']:.0f}°C max | util {s['ut_avg']:.0f}%")
    print(f"{indent}降频占比: {thr_s}"
          + ("  (SwPowerCap 满载正常)" if 'SwPowerCap' in thr else ""))
    if s['ecc'] is not None:
        print(f"{indent}ECC 不可纠正错误: {s['ecc']}")
        if s['ecc'] > 0:
            warns.append(f'ECC 不可纠正错误 {s["ecc"]} 个')
    if check_load and s['pw_avg'] < 400:
        warns.append(f'平均功耗仅 {s["pw_avg"]:.0f}W (<400W 目标), 未压满或已降频')
    if check_load and s['ut_avg'] < 90:
        warns.append(f'GPU 利用率仅 {s["ut_avg"]:.0f}% (<90%)')

    kind_of = {n: k for _, n, k in THROTTLE_FLAGS}
    hw = {k: v for k, v in thr.items() if kind_of.get(k) == 'hw'}
    th = {k: v for k, v in thr.items() if kind_of.get(k) == 'thermal'}
    if hw:
        warns.append('硬件级降频: ' + ' '.join(f'{k}={v*100:.0f}%' for k, v in hw.items()))
    if th:
        worst = max(th.values())
        # 偶发几个采样点无所谓, 长时间热降频才是实际损失吞吐的
        level = '⚠️ ' if worst >= 0.05 else ''
        msg = (f'{level}热降频 {worst*100:.0f}% 时间 ('
               + ' '.join(f'{k}={v*100:.0f}%' for k, v in sorted(th.items(), key=lambda x: -x[1]))
               + ') -> 散热不足, 非显卡故障')
        if worst >= 0.05:
            warns.append(msg)
        else:
            print(f"{indent}{msg}")
    if s['tp_avg'] >= TEMP_TARGET:
        warns.append(f'平均温度 {s["tp_avg"]:.1f}°C 已达降频线 {TEMP_TARGET}°C, 全程受热限制')
    elif s['tp_max'] >= TEMP_SLOWDOWN:
        warns.append(f'温度峰值 {s["tp_max"]:.0f}°C 触及硬件降频线 {TEMP_SLOWDOWN}°C')
    if s['tp_max'] >= TEMP_SHUTDOWN - 1:
        warns.append(f'温度峰值 {s["tp_max"]:.0f}°C 逼近保护关机线 {TEMP_SHUTDOWN}°C, 需处理散热')
    return warns


def report_single(gpu_id, phys_id, res, mon_summary, duration_min):
    print(f"\n{'='*74}")
    print(f"  完成时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  运行时长: {res['minutes']:.2f} 分钟")
    print(f"  总步数: {res['steps']}   吞吐量: {res['throughput']:.1f} steps/min")
    print(f"  数值校验: {res['checks']} 轮")
    if res['mem_pct']:
        print(f"  显存占用: {res['mem_pct']:.0f}%")
    print(f"\n  --- 硬件状态 (物理 GPU {phys_id}) ---")
    if duration_min <= 0:
        print("  (仅单元扫描, 功耗/利用率不作达标判断)")
    warns = print_hw(mon_summary, phys_id, check_load=duration_min > 0)

    if res['bit_warns']:
        w = res['bit_warns']
        print(f"\n  ⚠️  {len(w)}/{res['checks']} 轮出现结果非 bit-wise 一致 (精度仍达标):")
        for t, k, d, n in w[:5]:
            print(f"     [{t:.1f}min] {k}: {n} 个元素不同, 最大偏差 {d:.2e}")
        print(f"     同卡同 shape 本应逐位可复现; 若反复出现需怀疑计算单元不稳定")
        warns.append(f'{len(w)} 轮结果非 bit-wise 一致')

    xids, ok = check_xid()
    if not ok:
        print(f"\n  dmesg 不可读 (需要 root), 请手动: dmesg | grep -i xid | tail -10")
    elif xids:
        print(f"\n  !! dmesg 中发现 Xid ({len(xids)} 条, 最近几条):")
        for l in xids[-5:]:
            print(f"     {l[:110]}")
        warns.append(f'dmesg 中存在 Xid 记录 {len(xids)} 条')
    else:
        print(f"\n  dmesg: 无 Xid 记录")

    print(f"\n{'='*74}")
    if res['errors']:
        print(f"  ❌ GPU {gpu_id} 检出 {len(res['errors'])} 个错误:")
        for t, phase, msg in res['errors'][:10]:
            print(f"     [{t:.1f}min] {phase}: {msg[:100]}")
        print(f"\n  结论: GPU {gpu_id} 存在故障")
        print(f"  检查: dmesg | grep -i xid | tail -20")
    else:
        print(f"  ✅ GPU {gpu_id} 在 {res['minutes']:.1f}min 压力下无计算错误、无数值偏差")
        if res['minutes'] < duration_min * 0.95 and duration_min > 0:
            print(f"     (未跑满 {duration_min}min)")
    if warns:
        print(f"\n  ⚠️  {len(warns)} 项告警:")
        for w in warns:
            print(f"     - {w}")
    print(f"{'='*74}\n")


def stress_all_gpus(duration_min=30, skip_sweep=False):
    """所有GPU并行压力测试"""
    import concurrent.futures
    import threading

    num_gpus = torch.cuda.device_count()
    pmap = phys_index_map()

    print(f"\n{'='*74}")
    print(f"  全卡并行高强度压力测试 + 数值校验")
    print(f"  GPU数量: {num_gpus}   持续时间: {duration_min} 分钟")
    print(f"{'='*74}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i} (物理 {pmap[i]}): {props.name} | {props.total_memory/1024**3:.1f}GB")

    monitor = GPUMonitor([pmap[i] for i in range(num_gpus)])
    monitor.start()

    start = time.time()
    end_time = start + duration_min * 60
    print(f"\n  开始时间: {datetime.fromtimestamp(start).strftime('%H:%M:%S')}")
    print(f"  预计结束: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print(f"  运行中 (前约15秒为单元扫描)...\n")

    results = {}
    stop_monitor = threading.Event()

    def monitor_progress():
        last_print = start
        while not stop_monitor.is_set():
            now = time.time()
            if now - last_print >= 5.0:
                elapsed = now - start
                remaining = max(0, duration_min * 60 - elapsed)
                percent = min(100, int(elapsed / max(1, duration_min * 60) * 100))
                print(f"  [{percent:3d}%] 已运行 {int(elapsed//60)}:{int(elapsed%60):02d} | "
                      f"剩余 {int(remaining//60)}:{int(remaining%60):02d} | "
                      f"完成 {len(results)}/{num_gpus} GPU")
                last_print = now
            time.sleep(1)

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(stress_test_heavy, gid, duration_min, True, skip_sweep): gid
                       for gid in range(num_gpus)}
            for future in concurrent.futures.as_completed(futures):
                gid = futures[future]
                try:
                    results[gid] = future.result()
                except Exception as e:
                    results[gid] = dict(steps=0, minutes=0.0, throughput=0.0,
                                        errors=[(0.0, '线程异常', str(e))], bit_warns=[],
                                        checks=0, sweep=[], mem_pct=0.0)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        monitor.stop()

    mon = monitor.summary()
    total_time = time.time() - start
    print(f"\n  实际用时: {int(total_time//60)}:{int(total_time%60):02d}")

    print(f"\n{'='*74}")
    print("  汇总结果:")
    print(f"{'='*74}")
    all_warns = {}
    for gid in sorted(results.keys()):
        r = results[gid]
        status = "❌" if r['errors'] else "✅"
        print(f"\n  {status} GPU {gid} (物理 {pmap[gid]}): {r['steps']} steps / "
              f"{r['minutes']:.2f}min ({r['throughput']:.1f} steps/min) | "
              f"校验 {r['checks']} 轮 | 显存 {r['mem_pct']:.0f}%")
        w = print_hw(mon, pmap[gid], indent='      ')
        if r['errors']:
            for t, phase, msg in r['errors'][:5]:
                print(f"      错误 [{t:.1f}min] {phase}: {msg[:90]}")
        if r['bit_warns']:
            bw = r['bit_warns']
            print(f"      非 bit-wise 一致 {len(bw)}/{r['checks']} 轮, "
                  f"最大偏差 {max(x[2] for x in bw):.2e}")
            w.append(f'{len(bw)} 轮结果非 bit-wise 一致')
        if w:
            all_warns[gid] = w
            for x in w:
                print(f"      ⚠️  {x}")
    monitor.cleanup()

    # 吞吐量横向对比
    tps = {g: r['throughput'] for g, r in results.items() if r['throughput'] > 0}
    if len(tps) > 1:
        vals = list(tps.values())
        avg = sum(vals) / len(vals)
        print(f"\n  吞吐量: avg={avg:.1f} min={min(vals):.1f} max={max(vals):.1f} steps/min")
        slow = [g for g, t in tps.items() if t < avg * 0.7]
        if slow:
            print(f"  ⚠️  GPU {slow} 吞吐量低于平均 70%, 可能有硬件问题")

    # 单元扫描横向对比: 同型号卡之间某个单元明显慢, 说明该单元异常
    sweeps = {g: r['sweep'] for g, r in results.items() if r['sweep']}
    if len(sweeps) > 1:
        print(f"\n  单元吞吐横向对比 (TFLOPS, 低于均值 80% 标 !):")
        names = [x[0] for x in next(iter(sweeps.values()))]
        for i, name in enumerate(names):
            vals = {}
            for g, s in sweeps.items():
                if i < len(s) and s[i][1] == s[i][1]:
                    vals[g] = s[i][1]
            if len(vals) < 2:
                continue
            avg = sum(vals.values()) / len(vals)
            bad = [g for g, v in vals.items() if avg > 0 and v < avg * 0.8]
            line = ' '.join(f"G{g}:{v:.0f}" for g, v in sorted(vals.items()))
            print(f"   {'!' if bad else ' '}{name:<28} avg {avg:6.1f} | {line}"
                  + (f"  <- GPU {bad} 偏低" if bad else ""))

    xids, ok = check_xid()
    if ok and xids:
        print(f"\n  !! dmesg 中发现 Xid ({len(xids)} 条):")
        for l in xids[-8:]:
            print(f"     {l[:110]}")
    elif ok:
        print(f"\n  dmesg: 无 Xid 记录")
    else:
        print(f"\n  dmesg 不可读, 请手动: dmesg | grep -i xid | tail -20")

    fail = [g for g, r in results.items() if r['errors']]
    print(f"\n{'='*74}")
    if fail:
        print(f"  ❌ GPU {fail} 检出错误")
    else:
        total_steps = sum(r['steps'] for r in results.values())
        total_chk = sum(r['checks'] for r in results.values())
        print(f"  ✅ 全部 {num_gpus} 卡在 {duration_min}min 压力下无错误 "
              f"(共 {total_steps} steps, {total_chk} 轮数值校验)")
    if all_warns:
        print(f"  ⚠️  GPU {sorted(all_warns)} 有告警, 见上方明细")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU 高强度压力测试 + 计算单元扫描 + 数值正确性校验 + 硬件监控")
    parser.add_argument("--gpu", type=int, default=0, help="单卡测试的GPU ID")
    parser.add_argument("--duration", type=int, default=30,
                        help="持续时间(分钟), 0=只跑单元扫描(约15秒)")
    parser.add_argument("--all", action="store_true", help="测试所有GPU")
    parser.add_argument("--skip-sweep", action="store_true", help="跳过计算单元扫描")
    args = parser.parse_args()

    print(f"{'#'*74}")
    print(f"#  GPU 压力测试与硬件诊断  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {os.uname().nodename}")
    print(f"#  PyTorch={torch.__version__} CUDA={torch.version.cuda} "
          f"cuDNN={torch.backends.cudnn.version()} 可见GPU={torch.cuda.device_count()}")
    print(f"{'#'*74}")

    if args.all:
        stress_all_gpus(args.duration, skip_sweep=args.skip_sweep)
    else:
        pmap = phys_index_map()
        phys = pmap.get(args.gpu, args.gpu)
        monitor = GPUMonitor([phys])
        monitor.start()
        try:
            res = stress_test_heavy(args.gpu, args.duration, skip_sweep=args.skip_sweep)
        finally:
            monitor.stop()
        report_single(args.gpu, phys, res, monitor.summary(), args.duration)
        monitor.cleanup()

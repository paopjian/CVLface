"""编排内自检: 前 5 个 off-diag tile 的 fused_he 命中对逐对现算核对"""
import os
import sys

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import sklearn.preprocessing
import torch
import torch.multiprocessing as mp

import eval_all_trt_single as E
from evaluations.cluster_utils import _get_v7_he_ext

DATA = '/data1/dataset_0605/test_enhance'


def main():
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    shm = '/dev/shm/selfcheck'
    os.makedirs(shm, exist_ok=True)
    procs = []
    for rank in range(7):
        p = mp.Process(target=E.worker_extract,
                       args=(rank, 7, '/tmp/int8_ptq_work/engine_fp16.plan', DATA, shm))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    f_n, f_f, labels = E.gather_and_deduplicate(shm, 7)
    import shutil
    shutil.rmtree(shm, ignore_errors=True)
    emb = sklearn.preprocessing.normalize((f_n + f_f).numpy())
    codes = np.unique(labels.numpy(), return_inverse=True)[1].astype(np.int32)
    N = len(ids := labels.numpy())
    BLK = 16384
    nb = (N + BLK - 1) // BLK
    ext = _get_v7_he_ext()

    feats = torch.from_numpy(emb).half()
    try:
        torch.cuda.cudart().cudaHostRegister(feats.data_ptr(), feats.nbytes, 0)
    except Exception:
        feats = feats.pin_memory()
    code_t = torch.from_numpy(codes).pin_memory()
    dev = torch.device('cuda:0')
    shard = {}
    for bj in range(nb):
        shard[bj] = (feats[bj * BLK:min((bj + 1) * BLK, N)].to(dev),
                     code_t[bj * BLK:min((bj + 1) * BLK, N)].to(dev))

    mismatch = False
    for bi, bj in [(0, 1), (0, 2), (1, 2), (2, 3), (0, 7)]:
        r0, r1 = bi * BLK, min((bi + 1) * BLK, N)
        c0, c1 = bj * BLK, min((bj + 1) * BLK, N)
        C = c1 - c0
        R = r1 - r0
        blk1 = feats[r0:r1].to(dev)
        cd1 = code_t[r0:r1].to(dev)
        blk2, cd2 = shard[bj]
        sim = torch.matmul(blk1, blk2.T)
        hist32 = torch.zeros(2000, device=dev, dtype=torch.int32)
        ni = torch.empty(100000, device=dev, dtype=torch.int32)
        nj = torch.empty(100000, device=dev, dtype=torch.int32)
        ns = torch.empty(100000, device=dev, dtype=torch.float32)
        ncnt = torch.zeros(1, device=dev, dtype=torch.int32)
        dum = torch.empty(100000, device=dev, dtype=torch.int32)
        dumf = torch.empty(100000, device=dev, dtype=torch.float32)
        cnt0 = torch.zeros(1, device=dev, dtype=torch.int32)
        ext.fused_he(sim, C, cd1, cd2, -1.0, 1.0, 1000.0, 2000, 0.5, -2.0, r0, c0,
                     100000, hist32, ni, nj, ns, ncnt, dum, dum, dumf, cnt0, 4096, 128)
        nsz = int(ncnt.item())
        gi = ni[:nsz].cpu().numpy()
        gj = nj[:nsz].cpu().numpy()
        gs = ns[:nsz].cpu().numpy()
        s32 = sim.float().cpu()
        bad = 0
        for k in range(min(3, nsz)):
            i, j, sv = int(gi[k]), int(gj[k]), float(gs[k])
            li, lj = i - r0, j - c0
            real = float(emb[i] @ emb[j])
            tile_val = float(s32[li, lj]) if 0 <= li < R and 0 <= lj < C else float('nan')
            ok = abs(real - sv) < 0.01
            if not ok:
                bad = True
                mismatch = True
            print(f"  tile({bi},{bj}) 对{k}: ({i},{j}) 核 {sv:.4f} "
                  f"现算 {real:.4f} tile内 {tile_val:.4f} {'OK' if ok else 'MISMATCH'}")
        del sim, hist32, ni, nj, ns, ncnt, blk1, cd1
        torch.cuda.empty_cache()
    print("结论:", "编排内复现错配" if mismatch else "编排内核结果全部正确")


if __name__ == '__main__':
    main()

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// v7 单遍融合核：fp16 直读 + full/pos 双 smem 桶 + 双阈值样本对提取, 一次读全做。
// 骨架=55 号 histpn (双桶+步进增量坐标+IS_DIAG 严格上三角), 提取=54 号
// (neg: 跨档案且 v>=thr_neg; pos: 同档案且 v<=thr_pos)。
// 坐标: gi = i + row_off, gj = j + col_off (全局行号)。
__global__ void fused_he_pn_kernel(
    const __half* __restrict__ x, long long n, int C,
    const int* __restrict__ rcd, const int* __restrict__ ccd,
    float lo, float hi, float invw, int bins,
    float thr_neg, float thr_pos, int is_diag,
    int row_off, int col_off, long long max_pairs,
    int* __restrict__ out_full, int* __restrict__ out_pos,
    int* __restrict__ ni, int* __restrict__ nj, float* __restrict__ ns,
    int* __restrict__ ncnt,
    int* __restrict__ pi, int* __restrict__ pj, float* __restrict__ ps,
    int* __restrict__ pcnt)
{
    extern __shared__ int sh[];
    int* shf = sh;
    int* shp = sh + bins;
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        shf[i] = 0;
        shp[i] = 0;
    }
    __syncthreads();
    long long stride = (long long)gridDim.x * blockDim.x;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int i = (int)(idx / C);
    int j = (int)(idx - (long long)i * C);
    int si = (int)(stride / C);
    int sj = (int)(stride - (long long)si * C);
    for (; idx < n; idx += stride) {
        float v = __half2float(x[idx]);
        int tri = (!is_diag) || (j > i);
        if (tri && v >= lo && v <= hi) {
            int b = (int)((v - lo) * invw);
            b = max(0, min(b, bins - 1));
            int ci = rcd[i], cj = ccd[j];
            atomicAdd(&shf[b], 1);
            if (ci == cj) atomicAdd(&shp[b], 1);
            if (v >= thr_neg && ci != cj) {
                int s = atomicAdd(ncnt, 1);
                if (s < (int)max_pairs) {
                    ni[s] = i + row_off; nj[s] = j + col_off; ns[s] = v;
                }
            } else if (v <= thr_pos && ci == cj) {
                int s = atomicAdd(pcnt, 1);
                if (s < (int)max_pairs) {
                    pi[s] = i + row_off; pj[s] = j + col_off; ps[s] = v;
                }
            }
        }
        i += si;
        j += sj;
        if (j >= C) {
            j -= C;
            i += 1;
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        if (shf[b]) atomicAdd(&out_full[b], shf[b]);
        if (shp[b]) atomicAdd(&out_pos[b], shp[b]);
    }
}

void fused_he_pn(torch::Tensor x, long C, torch::Tensor rcd, torch::Tensor ccd,
                 double lo, double hi, double invw, long bins,
                 double thr_neg, double thr_pos, long is_diag,
                 long row_off, long col_off, long max_pairs,
                 torch::Tensor out_full, torch::Tensor out_pos,
                 torch::Tensor ni, torch::Tensor nj, torch::Tensor ns,
                 torch::Tensor ncnt,
                 torch::Tensor pi, torch::Tensor pj, torch::Tensor ps,
                 torch::Tensor pcnt, long blocks, long threads) {
    auto x_c = x.contiguous();
    long long n = x_c.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_he_pn_kernel<<<(int)blocks, (int)threads, 2 * bins * sizeof(int),
                         stream>>>(
        (const __half*)x_c.data_ptr(), n, (int)C,
        rcd.data_ptr<int>(), ccd.data_ptr<int>(),
        (float)lo, (float)hi, (float)invw, (int)bins,
        (float)thr_neg, (float)thr_pos, (int)is_diag,
        (int)row_off, (int)col_off, max_pairs,
        out_full.data_ptr<int>(), out_pos.data_ptr<int>(),
        ni.data_ptr<int>(), nj.data_ptr<int>(), ns.data_ptr<float>(),
        ncnt.data_ptr<int>(),
        pi.data_ptr<int>(), pj.data_ptr<int>(), ps.data_ptr<float>(),
        pcnt.data_ptr<int>());
}

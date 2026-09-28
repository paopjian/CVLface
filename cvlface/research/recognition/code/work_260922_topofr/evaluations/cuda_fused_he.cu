#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// 融合核（仅 off-diag tile）：fp16 单遍直读
//   1) 全量直方图 @bins（smem 私有化，块内原子，收尾合并非零 bin）
//   2) neg 提取：跨档案 & v>=thr_neg（值过滤先行，命中才算坐标/查码）
//   3) pos 提取：同档案 & v<thr_pos
// 索引：gi = idx/C + row_off，gj = idx%C + col_off（全局排序序）
__global__ void fused_he_kernel(
    const __half* __restrict__ x, long long n, int C,
    const int* __restrict__ rcd, const int* __restrict__ ccd,
    float lo, float hi, float invw, int bins,
    float thr_neg, float thr_pos,
    int row_off, int col_off, long long max_pairs,
    int* __restrict__ hist,
    int* __restrict__ ni, int* __restrict__ nj, float* __restrict__ ns,
    int* __restrict__ ncnt,
    int* __restrict__ pi, int* __restrict__ pj, float* __restrict__ ps,
    int* __restrict__ pcnt)
{
    extern __shared__ int sh[];
    for (int i = threadIdx.x; i < bins; i += blockDim.x) sh[i] = 0;
    __syncthreads();
    long long stride = (long long)gridDim.x * blockDim.x;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += stride) {
        float v = __half2float(x[idx]);
        if (v >= lo && v <= hi) {
            int b = (int)((v - lo) * invw);
            b = max(0, min(b, bins - 1));
            atomicAdd(&sh[b], 1);
        }
        if (v >= thr_neg || v < thr_pos) {
            int i = (int)(idx / C);
            int j = (int)(idx - (long long)i * C);
            int ci = rcd[i], cj = ccd[j];
            int gi = i + row_off, gj = j + col_off;
            if (v >= thr_neg && ci != cj) {
                int s = atomicAdd(ncnt, 1);
                if (s < (int)max_pairs) { ni[s] = gi; nj[s] = gj; ns[s] = v; }
            } else if (v < thr_pos && ci == cj) {
                int s = atomicAdd(pcnt, 1);
                if (s < (int)max_pairs) { pi[s] = gi; pj[s] = gj; ps[s] = v; }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
        if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

void fused_he(torch::Tensor x, long C, torch::Tensor rcd, torch::Tensor ccd,
              double lo, double hi, double invw, long bins,
              double thr_neg, double thr_pos,
              long row_off, long col_off, long max_pairs,
              torch::Tensor hist,
              torch::Tensor ni, torch::Tensor nj, torch::Tensor ns,
              torch::Tensor ncnt,
              torch::Tensor pi, torch::Tensor pj, torch::Tensor ps,
              torch::Tensor pcnt, long blocks, long threads) {
    auto x_c = x.contiguous();
    long long n = x_c.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_he_kernel<<<(int)blocks, (int)threads, bins * sizeof(int), stream>>>(
        (const __half*)x_c.data_ptr(), n, (int)C,
        rcd.data_ptr<int>(), ccd.data_ptr<int>(),
        (float)lo, (float)hi, (float)invw, (int)bins,
        (float)thr_neg, (float)thr_pos,
        (int)row_off, (int)col_off, max_pairs,
        hist.data_ptr<int>(),
        ni.data_ptr<int>(), nj.data_ptr<int>(), ns.data_ptr<float>(),
        ncnt.data_ptr<int>(),
        pi.data_ptr<int>(), pj.data_ptr<int>(), ps.data_ptr<float>(),
        pcnt.data_ptr<int>());
}

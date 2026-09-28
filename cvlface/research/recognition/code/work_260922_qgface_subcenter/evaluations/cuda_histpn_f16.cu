#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// fp16 直读双桶直方图核：full（全体）+ pos（同档案），neg 下游 = full - pos。
// 单遍读 fp16，寄存器转 fp32，分桶公式与 torch.histc 同款；smem 双桶私有化。
// diag tile（is_diag=1）：仅严格上三角（局部 j > i）。坐标用步进增量
//（避免逐元素除法）：idx 每步 +stride ⟹ i += si, j += sj（sj<C，至多进位一次）。
__global__ void histpn_f16_kernel(
    const __half* __restrict__ x, long long n, int C,
    const int* __restrict__ rcd, const int* __restrict__ ccd,
    float lo, float hi, float invw, int bins, int is_diag,
    int* __restrict__ out_full, int* __restrict__ out_pos)
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
            atomicAdd(&shf[b], 1);
            if (rcd[i] == ccd[j]) atomicAdd(&shp[b], 1);
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

void histpn_f16(torch::Tensor x, long C, torch::Tensor rcd, torch::Tensor ccd,
                double lo, double hi, double invw, long bins, long is_diag,
                torch::Tensor out_full, torch::Tensor out_pos, long blocks,
                long threads) {
    auto x_c = x.contiguous();
    long long n = x_c.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    histpn_f16_kernel<<<(int)blocks, (int)threads, 2 * bins * sizeof(int),
                        stream>>>(
        (const __half*)x_c.data_ptr(), n, (int)C,
        rcd.data_ptr<int>(), ccd.data_ptr<int>(),
        (float)lo, (float)hi, (float)invw, (int)bins, (int)is_diag,
        out_full.data_ptr<int>(), out_pos.data_ptr<int>());
}

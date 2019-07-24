#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void relToAbsIndex_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> relIndx,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> init_spIndx,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> absIndx,
    int batch_size, int length, int height, int width, int Kl, int Kh, int Kw, int K) {
    // indexing
    const int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = height * width;
    const int l = d / HW;
    d = d % HW;
    const int h = d / width;
    const int w = d % width;
    if (l < length) {
        const int r_idx = relIndx[n][0][l][h][w];
        const int r_idx_l = r_idx / 9 - 1;
        int r_idx_h = (r_idx % 9) / 3 - 1;
        int r_idx_w = (r_idx % 9) % 3 - 1;

        const int init_spix_index = init_spIndx[n][0][l][h][w];
        const int Khw = Kh * Kw;

        int spix_idx_l = init_spix_index + r_idx_l * Khw;
        if (spix_idx_l >= K || spix_idx_l <= -1) {
            spix_idx_l = init_spix_index;
        }

        if (((spix_idx_l + Kw) % Khw) == 0 && r_idx_h == 1) {
            r_idx_h = 0;
        } else if ((spix_idx_l % Khw) == 0 && r_idx_h == -1) {
            r_idx_h = 0;
        }
        int spix_idx_h = spix_idx_l + r_idx_h * Kw;
        if (spix_idx_h >= K || spix_idx_h <= -1) {
            spix_idx_h = spix_idx_l;
        }

        if (((spix_idx_h + 1) % Kw) == 0 && r_idx_w == 1) {
            r_idx_w = 0;
        } else if ((spix_idx_h % Kw) == 0 && r_idx_w == -1) {
            r_idx_w = 0;
        }
        int spix_idx_w = spix_idx_h + r_idx_w;
        if (spix_idx_w < K && spix_idx_w > -1) {
            absIndx[n][0][l][h][w] = float(spix_idx_w);
        } else {
            absIndx[n][0][l][h][w] = float(spix_idx_h);
        }
    }
}
} // namespace

torch::Tensor relToAbsIndex_cuda_forward(
    const torch::Tensor relIndx,  // B 1 L H W
    const torch::Tensor init_spIndx,  // B 1 L H W
    const int Kl,
    const int Kh,
    const int Kw) {
    // setup
    const auto batch_size = relIndx.size(0);
    const auto length = relIndx.size(2);
    const auto height  = relIndx.size(3);
    const auto width  = relIndx.size(4);
    auto absIndx = torch::zeros_like(relIndx);  // B 1 H W
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((length * height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(relIndx.type(), "relToAbsIndex_forward_cuda", ([&] {
        relToAbsIndex_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            relIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            absIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            batch_size, length, height, width, Kl, Kh, Kw, Kl*Kh*Kw);
    }));

    return absIndx;
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void relToAbsIndex2d_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> relIndx,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> init_spIndx,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> absIndx,
    int height, int width, int Kh, int Kw, int K) {
    // indexing
    const int n = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = d / width;
    const int w = d % width;
    if (h < height) {
        // Convert spix_idx based on the rel_idx
        const int rel_idx = static_cast<int>(relIndx[n][0][h][w]);
        const int rel_idx_h = rel_idx / 3 - 1;
        int rel_idx_w = rel_idx % 3 - 1;

        const int init_spix_idx = static_cast<int>(init_spIndx[n][0][h][w]);
        int spix_idx_h = init_spix_idx + rel_idx_h * Kw;
        if (spix_idx_h >= K || spix_idx_h <= -1) {
            spix_idx_h = init_spix_idx;
        }

        if (((spix_idx_h + 1) % Kw) == 0 && rel_idx_w == 1) {
            rel_idx_w = 0;
        } else if ((spix_idx_h % Kw) == 0 && rel_idx_w == -1) {
            rel_idx_w = 0;
        }
        int spix_idx_w = spix_idx_h + rel_idx_w;
        if (spix_idx_w < K && spix_idx_w > -1) {
            absIndx[n][0][h][w] = static_cast<float>(spix_idx_w);
        } else {
            absIndx[n][0][h][w] = static_cast<float>(spix_idx_h);
        }
    }
}
} // namespace

torch::Tensor relToAbsIndex2d_cuda_forward(
    const torch::Tensor relIndx,  // B 1 H W
    const torch::Tensor init_spIndx,  // B 1 H W
    const int Kh,
    const int Kw) {
    // setup
    const auto batch_size = relIndx.size(0);
    const auto height = relIndx.size(2);
    const auto width  = relIndx.size(3);
    auto absIndx = torch::zeros_like(relIndx);  // B 1 H W
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(relIndx.type(), "relToAbsIndex2d_forward_cuda", ([&] {
        relToAbsIndex2d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            relIndx.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            absIndx.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            height, width, Kh, Kw, Kh*Kw);
    }));

    return absIndx;
}
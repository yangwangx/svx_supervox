#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static inline  __device__  void atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
#endif

namespace {
template <typename scalar_t>
__global__ void spFeatSmear2d_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> spIndx,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> pFeat,
    int batch_size, int depth, int height, int width, int K) {
    // indexing
    const int n = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = d / width;
    const int w = d % width;
    const int spix_idx = static_cast<int>(spIndx[n][0][h][w]);
    if (h < height) {
        for (int k = 0; k < depth; k++) {
            pFeat[n][k][h][w] = spFeat[n][k][spix_idx];
        }
    }
}

template <typename scalar_t>
__global__ void spFeatSmear2d_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_pFeat,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> spIndx,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_spFeat,
    int batch_size, int depth, int height, int width) {
    // indexing
    const int n = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = d / width;
    const int w = d % width;
    const int spix_idx = static_cast<int>(spIndx[n][0][h][w]);
    
    if (h < height) {
        for (int k = 0; k < depth; k++) {
            atomicAdd(&grad_spFeat[n][k][spix_idx], grad_pFeat[n][k][h][w]);
        }
    }
}
} // namespace

torch::Tensor spFeatSmear2d_cuda_forward(
    const torch::Tensor spFeat,  // B C K
    const torch::Tensor spIndx) {  // B 1 H W
    // setup
    const auto batch_size = spFeat.size(0);
    const auto depth = spFeat.size(1);
    const auto K = spFeat.size(2);
    const auto height = spIndx.size(2);
    const auto width  = spIndx.size(3);
    auto pFeat = torch::zeros({batch_size, depth, height, width},
        torch::TensorOptions().dtype(spFeat.dtype()).device(spFeat.device()).requires_grad(true));  // B C H W
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "spFeatSmear2d_forward_cuda", ([&] {
        spFeatSmear2d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spIndx.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            pFeat.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            depth, height, width, K);
    }));

    return pFeat;
}


torch::Tensor spFeatSmear2d_cuda_backward(
    const torch::Tensor grad_pFeat,  // B C H W
    const torch::Tensor spIndx,  // B 1 H W
    const int K) {
    // setup
    const auto batch_size = grad_pFeat.size(0);
    const auto depth = grad_pFeat.size(1);
    const auto height = grad_pFeat.size(2);
    const auto width  = grad_pFeat.size(3);
    auto grad_spFeat = torch::zeros({batch_size, depth, K},
        torch::TensorOptions().dtype(grad_pFeat.dtype()).device(grad_pFeat.device()).requires_grad(false));  // B C 1 K
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(grad_pFeat.type(), "spFeatSmear2d_backward_cuda", ([&] {
        spFeatSmear2d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_pFeat.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            spIndx.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            grad_spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            depth, height, width);
    }));

    return grad_spFeat;
}
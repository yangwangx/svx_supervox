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
__global__ void spFeat_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pFeat,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> init_spIndx,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spWght,
    int depth, int length, int height, int width, int K, int ignore_idx_value) {
    // indexing
    const int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = height * width;
    const int l = d / HW;
    d %= HW;
    const int h = d / width;
    const int w = d % width;
    if (l < length) {
        const int spixel_idx = static_cast<int>(init_spIndx[n][0][l][h][w]);
        if (spixel_idx != ignore_idx_value) {
            for (int k = 0; k < depth; k++) {
                atomicAdd(&spFeat[n][k][spixel_idx], pFeat[n][k][l][h][w]);
            }
            atomicAdd(&spWght[n][spixel_idx], 1.0);
        }
    }
}

template <typename scalar_t>
__global__ void spFeat_normalize_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spWght,
    int depth, int K, float ignore_feature_value) {
    // indexing
    const int n = blockIdx.y;
    const int spix_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spix_idx < K) {
        bool zeroWght = (spWght[n][spix_idx] < 0.001);
        for (int k = 0; k < depth; k++) {
            if (zeroWght) {
                spFeat[n][k][spix_idx] = ignore_feature_value;
            } else {
                spFeat[n][k][spix_idx] /= spWght[n][spix_idx];
            }
        }
    }
}
} //namespace

std::vector<torch::Tensor> spFeat_cuda_forward(
    const torch::Tensor pFeat,  // B C L H W
    const torch::Tensor init_spIndx,  // B 1 L H W
    const int K,
    const int ignore_idx_value,
    const float ignore_feature_value) {
    // setup
    const auto batch_size = pFeat.size(0);
    const auto depth = pFeat.size(1);
    const auto length = pFeat.size(2);
    const auto height = pFeat.size(3);
    const auto width  = pFeat.size(4);
    auto spFeat = torch::zeros({batch_size, depth, K},
        torch::TensorOptions().dtype(pFeat.dtype()).device(pFeat.device()).requires_grad(false));  // B C K
    auto spWght = torch::zeros({batch_size, K},
        torch::TensorOptions().dtype(pFeat.dtype()).device(pFeat.device()).requires_grad(false));  // B K
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((length * height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(pFeat.type(), "spFeat_forward_cuda", ([&] {
        spFeat_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spWght.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, length, height, width, K, ignore_idx_value);
    }));
    const dim3 blocks2((K + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(pFeat.type(), "spFeat_normalize_forward_cuda", ([&] {
        spFeat_normalize_cuda_forward_kernel<scalar_t><<<blocks2, threads>>>(
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spWght.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, K, ignore_feature_value);
    }));

    return {spFeat, spWght};
}
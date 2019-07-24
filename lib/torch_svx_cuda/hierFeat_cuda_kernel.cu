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
__global__ void hierFeat_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spSize,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> assign,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> hierFeat,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> hierSize,
    int depth, int K) {
    // indexing
    const int n = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        const int hier_idx = static_cast<int>(assign[n][idx]);
        for (int k = 0; k < depth; k++) {
            atomicAdd(&hierFeat[n][k][hier_idx], spFeat[n][k][idx] * spSize[n][idx]);
        }
        atomicAdd(&hierSize[n][hier_idx], spSize[n][idx]);
    }
}

template <typename scalar_t>
__global__ void hierFeat_normalize_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> hierFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> hierSize,
    int depth, int hierK) {
    // indexing
    const int n = blockIdx.y;
    const int hier_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hier_idx < hierK) {
        bool zeroWght = (hierSize[n][hier_idx] < 0.001);
        for (int k = 0; k < depth; k++) {
            if (zeroWght) {
                hierFeat[n][k][hier_idx] = 0.0;
            } else {
                hierFeat[n][k][hier_idx] /= hierSize[n][hier_idx];
            }
        }
    }
}
} //namespace

std::vector<torch::Tensor> hierFeat_cuda_forward(
    const torch::Tensor spFeat,  // B C K
    const torch::Tensor spSize,  // B K
    const torch::Tensor assign,  // B K
    const int hierK) {
    // setup
    const auto batch_size = spFeat.size(0);
    const auto depth = spFeat.size(1);
    const auto K = spFeat.size(2);
    auto hierFeat = torch::zeros({batch_size, depth, hierK}, 
        torch::TensorOptions().dtype(spFeat.dtype()).device(spFeat.device()).requires_grad(false));  // B C P
    auto hierSize = torch::zeros({batch_size, hierK},
        torch::TensorOptions().dtype(spFeat.dtype()).device(spFeat.device()).requires_grad(false));  // B P
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((K + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "hierFeat_forward_cuda", ([&] {
        hierFeat_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spSize.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            assign.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            hierFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            hierSize.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, K);
    }));
    const dim3 blocks2((hierK + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "hierFeat_normalize_forward_cuda", ([&] {
        hierFeat_normalize_cuda_forward_kernel<scalar_t><<<blocks2, threads>>>(
            hierFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            hierSize.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, hierK);
    }));
    return {hierFeat, hierSize};
}
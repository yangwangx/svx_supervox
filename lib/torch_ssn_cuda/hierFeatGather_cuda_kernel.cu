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
__global__ void hierFeatGather_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spSize,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> assign,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> hierFeat,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> hierSize,
    int depth, int prevK) {
    // indexing
    const int n = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < prevK) {
        const int hier_idx = static_cast<int>(assign[n][idx]);
        for (int k = 0; k < depth; k++) {
            atomicAdd(&hierFeat[n][k][hier_idx], spFeat[n][k][idx] * spSize[n][idx]);
        }
        atomicAdd(&hierSize[n][hier_idx], spSize[n][idx]);
    }
}

template <typename scalar_t>
__global__ void hierFeatGather_normalize_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> hierFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> hierSize,
    int depth, int hierK) {
    // indexing
    const int n = blockIdx.y;
    const int hier_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hier_idx < hierK) {
        bool zeroSize = (hierSize[n][hier_idx] < 0.001);
        for (int k = 0; k < depth; k++) {
            if (zeroSize) {
                hierFeat[n][k][hier_idx] = 0.0;
            } else {
                hierFeat[n][k][hier_idx] /= hierSize[n][hier_idx];
            }
        }
    }
}
} //namespace

std::vector<torch::Tensor> hierFeatGather_cuda_forward(
    const torch::Tensor spFeat,  // B C prevK
    const torch::Tensor spSize,  // B prevK
    const torch::Tensor assign,  // B prevK
    const int hierK) {
    // setup
    const auto batch_size = spFeat.size(0);
    const auto depth = spFeat.size(1);
    const auto prevK = spFeat.size(2);
    auto hierFeat = torch::zeros({batch_size, depth, hierK},
        torch::TensorOptions().dtype(spFeat.dtype()).device(spFeat.device()).requires_grad(false));  // B C hierK
    auto hierSize = torch::zeros({batch_size, hierK},
        torch::TensorOptions().dtype(spFeat.dtype()).device(spFeat.device()).requires_grad(false));  // B hierK
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((prevK + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "hierFeatGather_forward_cuda", ([&] {
        hierFeatGather_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spSize.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            assign.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            hierFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            hierSize.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, prevK);
    }));
    const dim3 blocks2((hierK + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "hierFeatGather_normalize_forward_cuda", ([&] {
        hierFeatGather_normalize_cuda_forward_kernel<scalar_t><<<blocks2, threads>>>(
            hierFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            hierSize.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, hierK);
    }));

    return {hierFeat, hierSize};
}
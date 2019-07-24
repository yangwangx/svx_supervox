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
__global__ void spFeatUpdate_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pFeat,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> assoc,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> init_spIndx,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spWght,
    int batch_size, int depth, int length, int height, int width, int Kl, int Kh, int Kw, int K) {
    // indexing
    const int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = height * width;
    const int LHW = length * HW;
    const int c = d / LHW;
    d %= LHW;
    const int l = d / HW;
    d %= HW;
    const int h = d / width;
    const int w = d % width;
    const int init_spix_index = static_cast<int>(init_spIndx[n][0][l][h][w]);
    int spixel_idx = init_spix_index;
    if (c < 27) {
        // Convert spixel_idx based on the association channel
        const int rel_idx = c;
        const int rel_idx_l = rel_idx / 9 - 1;
        int rel_idx_h = (rel_idx % 9) / 3 - 1;
        int rel_idx_w = (rel_idx % 9) % 3 - 1;

        bool invalid_spixel = false;
        
        const int Khw = Kh * Kw;
        int spix_idx_l = init_spix_index + rel_idx_l * Khw;
        if (spix_idx_l >= K || spix_idx_l <= -1) {
            spix_idx_l = init_spix_index;
            invalid_spixel = true;
        }

        if (((spix_idx_l + Kw) % Khw) == 0 && rel_idx_h == 1) {
            rel_idx_h = 0;
            invalid_spixel = true;
        } else if ((spix_idx_l % Khw) == 0 && rel_idx_h == -1) {
            rel_idx_h = 0;
            invalid_spixel = true;
        }
        int spix_idx_h = spix_idx_l + rel_idx_h * Kw;
        if (spix_idx_h >= K || spix_idx_h <= -1) {
            spix_idx_h = spix_idx_l;
            invalid_spixel = true;
        }

        if (((spix_idx_h + 1) % Kw) == 0 && rel_idx_w == 1) {
            rel_idx_w = 0;
            invalid_spixel = true;
        } else if ((spix_idx_h % Kw) == 0 && rel_idx_w == -1) {
            rel_idx_w = 0;
            invalid_spixel = true;
        }
        int spix_idx_w = spix_idx_h + rel_idx_w;
        if (spix_idx_w < K && spix_idx_w > -1) {
            spixel_idx = spix_idx_w;
        } else {
            spixel_idx = spix_idx_h;
            invalid_spixel = true;
        }
        //
        if (invalid_spixel == false){
            for (int k = 0; k < depth; k++) {
                atomicAdd(&spFeat[n][k][spixel_idx], pFeat[n][k][l][h][w] * assoc[n][c][l][h][w]);
            }
            atomicAdd(&spWght[n][spixel_idx], assoc[n][c][l][h][w]);
        }
    }
}

template <typename scalar_t>
__global__ void spFeatUpdate_normalize_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spWght,
    int depth, int K) {
    // indexing
    const int n = blockIdx.y;
    const int spix_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spix_idx < K) {
        bool zeroWght = (spWght[n][spix_idx] < 0.001);
        for (int k = 0; k < depth; k++) {
            if (zeroWght) {
                spFeat[n][k][spix_idx] = 0;
            } else {
                spFeat[n][k][spix_idx] /= spWght[n][spix_idx];
            }
        }
    }
}

template <typename scalar_t>
__global__ void spFeatUpdate_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_spFeat,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> grad_pFeat,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> grad_assoc,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> spWght,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pFeat,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> assoc,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> init_spIndx,
    int depth, int length, int height, int width, int Kl, int Kh, int Kw, int K) {
    // indexing
    const int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;    
    const int HW = height * width;
    const int LHW = length * HW;
    const int c = d / LHW;
    d %= LHW;
    const int l = d / HW;
    d %= HW;
    const int h = d / width;
    const int w = d % width;
    const int init_spix_index = static_cast<int>(init_spIndx[n][0][l][h][w]);
    int spixel_idx = init_spix_index;
    if (c < 27) {
        // Convert spixel_idx based on the association channel
        const int rel_idx = c;
        const int rel_idx_l = rel_idx / 9 - 1;
        int rel_idx_h = (rel_idx % 9) / 3 - 1;
        int rel_idx_w = (rel_idx % 9) % 3 - 1;

        bool invalid_spixel = false;
        
        const int Khw = Kh * Kw;
        int spix_idx_l = init_spix_index + rel_idx_l * Khw;
        if (spix_idx_l >= K || spix_idx_l <= -1) {
            spix_idx_l = init_spix_index;
            invalid_spixel = true;
        }

        if (((spix_idx_l + Kw) % Khw) == 0 && rel_idx_h == 1) {
            rel_idx_h = 0;
            invalid_spixel = true;
        } else if ((spix_idx_l % Khw) == 0 && rel_idx_h == -1) {
            rel_idx_h = 0;
            invalid_spixel = true;
        }
        int spix_idx_h = spix_idx_l + rel_idx_h * Kw;
        if (spix_idx_h >= K || spix_idx_h <= -1) {
            spix_idx_h = spix_idx_l;
            invalid_spixel = true;
        }

        if (((spix_idx_h + 1) % Kw) == 0 && rel_idx_w == 1) {
            rel_idx_w = 0;
            invalid_spixel = true;
        } else if ((spix_idx_h % Kw) == 0 && rel_idx_w == -1) {
            rel_idx_w = 0;
            invalid_spixel = true;
        }
        int spix_idx_w = spix_idx_h + rel_idx_w;
        if (spix_idx_w < K && spix_idx_w > -1) {
            spixel_idx = spix_idx_w;
        } else {
            spixel_idx = spix_idx_h;
            invalid_spixel = true;
        }
        //
        if ( !invalid_spixel ) {
            bool nonzeroWght = (spWght[n][spixel_idx] > 0.001);
            for (int k = 0; k < depth; k++) {
                if ( nonzeroWght ) {
                    atomicAdd(&grad_pFeat[n][k][l][h][w], 
                        grad_spFeat[n][k][spixel_idx] * assoc[n][c][l][h][w] / spWght[n][spixel_idx]);
                    atomicAdd(&grad_assoc[n][c][l][h][w],
                        grad_spFeat[n][k][spixel_idx] * (pFeat[n][k][l][h][w] - spFeat[n][k][spixel_idx]) / spWght[n][spixel_idx]);
                }
            }
        }
    }
}
} // namespace

std::vector<torch::Tensor> spFeatUpdate_cuda_forward(
    const torch::Tensor pFeat,  // B C L H W
    const torch::Tensor assoc,  // B 27 L H W
    const torch::Tensor init_spIndx,  // B 1 L H W
    const int Kl,
    const int Kh, 
    const int Kw) {
    // setup
    const auto batch_size = pFeat.size(0);
    const auto depth = pFeat.size(1);
    const auto length = pFeat.size(2);
    const auto height = pFeat.size(3);
    const auto width  = pFeat.size(4);
    const int K = Kl * Kh * Kw;
    auto spFeat = torch::zeros({batch_size, depth, K}, 
        torch::TensorOptions().dtype(pFeat.dtype()).device(
            pFeat.device()).requires_grad(pFeat.requires_grad() || assoc.requires_grad()));  // B C K
    auto spWght = torch::zeros({batch_size, K},
        torch::TensorOptions().dtype(pFeat.dtype()).device(pFeat.device()).requires_grad(false));  // B K
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((27 * length * height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(pFeat.type(), "spFeatUpdate_forward_cuda", ([&] {
        spFeatUpdate_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            assoc.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spWght.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            batch_size, depth, length, height, width, Kl, Kh, Kw, K);
    }));
    const dim3 blocks2((K + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(pFeat.type(), "spFeatUpdate_normalize_forward_cuda", ([&] {
        spFeatUpdate_normalize_cuda_forward_kernel<scalar_t><<<blocks2, threads>>>(
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spWght.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            depth, K);
    }));

    return {spFeat, spWght};
}

std::vector<torch::Tensor> spFeatUpdate_cuda_backward(
    const torch::Tensor grad_spFeat,  // B C K
    const torch::Tensor spFeat,  // B C K
    const torch::Tensor spWght,  // B K
    const torch::Tensor pFeat,  // B C L H W
    const torch::Tensor assoc,  // B 27 L H W
    const torch::Tensor init_spIndx,  // B 1 L H W
    const int Kl,
    const int Kh,
    const int Kw) {
    // setup
    const auto batch_size = pFeat.size(0);
    const auto depth = pFeat.size(1);
    const auto length = pFeat.size(2);
    const auto height = pFeat.size(3);
    const auto width  = pFeat.size(4);
    const int K = Kl * Kh * Kw;
    auto grad_pFeat = torch::zeros_like(pFeat).set_requires_grad(false);
    auto grad_assoc = torch::zeros_like(assoc).set_requires_grad(false);
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((27 * length * height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "spFeatUpdate_backward_cuda", ([&] {
        spFeatUpdate_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            grad_assoc.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            spWght.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            assoc.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            depth, length, height, width, Kl, Kh, Kw, K);
    }));

    return {grad_pFeat, grad_assoc};
}
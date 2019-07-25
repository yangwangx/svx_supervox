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
__global__ void pspDist3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pFeat,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> init_spIndx,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> sqdist,
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
    const int init_spix_idx = static_cast<int>(init_spIndx[n][0][l][h][w]);
    int spix_idx = init_spix_idx;
    if (c < 27) {
        // Convert spix_idx based on the association channel
        const int rel_idx = c;
        const int rel_idx_l = rel_idx / 9 - 1;
        int rel_idx_h = (rel_idx % 9) / 3 - 1;
        int rel_idx_w = (rel_idx % 9) % 3 - 1;

        bool invalid_spixel = false;
        
        const int Khw = Kh * Kw;
        int spix_idx_l = init_spix_idx + rel_idx_l * Khw;
        if (spix_idx_l >= K || spix_idx_l <= -1) {
            spix_idx_l = init_spix_idx;
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
            spix_idx = spix_idx_w;
        } else {
            spix_idx = spix_idx_h;
            invalid_spixel = true;
        }
        // compute squared distance
        scalar_t sq_dist = 0;
        if (invalid_spixel) {
            sq_dist = 10000.0;
        } else {
          for (int k = 0; k < depth; k++) {
            sq_dist += pow(pFeat[n][k][l][h][w] - spFeat[n][k][spix_idx], 2);
          }
        }
        sqdist[n][c][l][h][w] = sq_dist;
    }
}

template <typename scalar_t>
__global__ void pspDist3d_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> grad_sqdist,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> grad_pFeat,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_spFeat,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pFeat,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> spFeat,
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
    const int init_spix_idx = static_cast<int>(init_spIndx[n][0][l][h][w]);
    int spix_idx = init_spix_idx;
    if (c < 27) {
        // Convert spix_idx based on the association channel
        const int rel_idx = c;
        const int rel_idx_l = rel_idx / 9 - 1;
        int rel_idx_h = (rel_idx % 9) / 3 - 1;
        int rel_idx_w = (rel_idx % 9) % 3 - 1;

        bool invalid_spixel = false;
        
        const int Khw = Kh * Kw;
        int spix_idx_l = init_spix_idx + rel_idx_l * Khw;
        if (spix_idx_l >= K || spix_idx_l <= -1) {
            spix_idx_l = init_spix_idx;
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
            spix_idx = spix_idx_w;
        } else {
            spix_idx = spix_idx_h;
            invalid_spixel = true;
        }
        //
        if ( !invalid_spixel ) {
            for (int k = 0; k < depth; k++) {
                scalar_t _grad_pFeat = grad_sqdist[n][c][l][h][w] * 2 * (pFeat[n][k][l][h][w] - spFeat[n][k][spix_idx]);
                atomicAdd(&grad_pFeat[n][k][l][h][w], _grad_pFeat);
                atomicAdd(&grad_spFeat[n][k][spix_idx], -_grad_pFeat);
            }
        }
    }
}
} // namespace

torch::Tensor pspDist3d_cuda_forward(
    const torch::Tensor pFeat,  // B C L H W
    const torch::Tensor spFeat,  // B C K
    const torch::Tensor init_spIndx,  // B 1 L H W
    const int Kl,
    const int Kh,
    const int Kw) {
    // setup
    const auto batch_size = pFeat.size(0);
    const auto depth = pFeat.size(1);
    const auto length = pFeat.size(2);
    const auto height  = pFeat.size(3);
    const auto width  = pFeat.size(4);
    const int K = Kl * Kh * Kw;
    auto sqdist = torch::zeros({batch_size, 27, length, height, width},
        torch::TensorOptions().dtype(pFeat.dtype()).device(pFeat.device()).requires_grad(true));  // B 27 L H W
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((27 * length * height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(pFeat.type(), "pspDist3d_forward_cuda", ([&] {
        pspDist3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            sqdist.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            depth, length, height, width, Kl, Kh, Kw, K);
    }));
    return sqdist;
}

std::vector<torch::Tensor> pspDist3d_cuda_backward(
    const torch::Tensor grad_sqdist,  // B 27 L H W
    const torch::Tensor pFeat,  // B C L H W
    const torch::Tensor spFeat,  // B C K
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
    auto grad_spFeat = torch::zeros_like(spFeat).set_requires_grad(false);
    // launch kernel
    const int threads = 1024;
    const dim3 blocks((27 * length * height * width + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(spFeat.type(), "pspDist3d_backward_cuda", ([&] {
        pspDist3d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_sqdist.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            grad_pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            grad_spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pFeat.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            spFeat.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            init_spIndx.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            depth, length, height, width, Kl, Kh, Kw, K);
    }));

    return {grad_pFeat, grad_spFeat};
}
#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// sqdist = pspDist3d_cuda.forward(pFeat, spFeat, init_spIndx, Kl, Kh, Kw)

torch::Tensor pspDist3d_cuda_forward(
    const torch::Tensor pFeat,
    const torch::Tensor spFeat,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw);  // return sqdist;

torch::Tensor pspDist3d_forward(
    const torch::Tensor pFeat,
    const torch::Tensor spFeat,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh, 
    const int Kw) {
    // check
    CHECK_INPUT(pFeat);
    CHECK_INPUT(spFeat);
    CHECK_INPUT(init_spIndx);
    // forward
    return pspDist3d_cuda_forward(pFeat, spFeat, init_spIndx, Kl, Kh, Kw);
}

/* Backward */
// grad_pFeat, grad_spFeat = pspDist3d_cuda.backward(grad_sqdist, sqdist, pFeat, spFeat, init_spIndx, Kl, Kh, Kw)

std::vector<torch::Tensor> pspDist3d_cuda_backward(
    const torch::Tensor grad_sqdist,
    const torch::Tensor pFeat,
    const torch::Tensor spFeat,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw);  // return {grad_pFeat, grad_spFeat};

std::vector<torch::Tensor> pspDist3d_backward(
    const torch::Tensor grad_sqdist,
    const torch::Tensor pFeat,
    const torch::Tensor spFeat,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw) {
    // check
    CHECK_INPUT(grad_sqdist);
    CHECK_INPUT(pFeat);
    CHECK_INPUT(spFeat);
    CHECK_INPUT(init_spIndx);
    // backward
    return pspDist3d_cuda_backward(grad_sqdist, pFeat, spFeat, init_spIndx, Kl, Kh, Kw);
}

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pspDist3d_forward, "pspDist3d forward (CUDA)");
    m.def("backward", &pspDist3d_backward, "pspDist3d backward (CUDA)");
}
#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// spFeat, spWght, = spFeatUpdate3d_cuda.forward(pFeat, assoc, init_spIndx, Kl, Kh, Kw)

std::vector<torch::Tensor> spFeatUpdate3d_cuda_forward(
    const torch::Tensor pFeat,
    const torch::Tensor assoc,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw);  // return {spFeat, spWght};

std::vector<torch::Tensor> spFeatUpdate3d_forward(
    const torch::Tensor pFeat,
    const torch::Tensor assoc,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw) {
    // check
    CHECK_INPUT(pFeat);
    CHECK_INPUT(assoc);
    CHECK_INPUT(init_spIndx);
    // forward
    return spFeatUpdate3d_cuda_forward(pFeat, assoc, init_spIndx, Kl, Kh, Kw);
}

/* Backward */
// grad_pFeat, grad_assoc = spFeatUpdate3d_cuda.backward(grad_spFeat, spFeat, spWght, Kh, Kw)

std::vector<torch::Tensor> spFeatUpdate3d_cuda_backward(
    const torch::Tensor grad_spFeat,
    const torch::Tensor spFeat,
    const torch::Tensor spWght,
    const torch::Tensor pFeat,
    const torch::Tensor assoc,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw);  // return {grad_pFeat, grad_assoc};

std::vector<torch::Tensor> spFeatUpdate3d_backward(
    const torch::Tensor grad_spFeat,
    const torch::Tensor spFeat,
    const torch::Tensor spWght,
    const torch::Tensor pFeat,
    const torch::Tensor assoc,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh, 
    const int Kw) {
    // check
    CHECK_INPUT(grad_spFeat);
    CHECK_INPUT(spFeat);
    CHECK_INPUT(spWght);
    CHECK_INPUT(pFeat);
    CHECK_INPUT(assoc);
    CHECK_INPUT(init_spIndx);
    // backward
    return spFeatUpdate3d_cuda_backward(grad_spFeat, spFeat, spWght, pFeat, assoc, init_spIndx, Kl, Kh, Kw);
}

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spFeatUpdate3d_forward, "spFeatUpdate3d forward (CUDA)");
    m.def("backward", &spFeatUpdate3d_backward, "spFeatUpdate3d backward (CUDA)");
}

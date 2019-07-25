#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// pFeat = spFeatSmear3d_cuda.forward(spFeat, spIndx)

torch::Tensor spFeatSmear3d_cuda_forward(
    const torch::Tensor spFeat,
    const torch::Tensor spIndx);  // return pFeat;

torch::Tensor spFeatSmear3d_forward(
    const torch::Tensor spFeat,
    const torch::Tensor spIndx) {
    // check
    CHECK_INPUT(spFeat);
    CHECK_INPUT(spIndx);
    // forward
    return spFeatSmear3d_cuda_forward(spFeat, spIndx);
}

/* Backward */
//  grad_spFeat = spFeatSmear3d_cuda.backward(grad_pFeat, spIndx, K)

torch::Tensor spFeatSmear3d_cuda_backward(
    const torch::Tensor grad_pFeat,
    const torch::Tensor spIndx,
    const int K);  // return grad_spFeat;

torch::Tensor spFeatSmear3d_backward(
    const torch::Tensor grad_pFeat,
    const torch::Tensor spIndx,
    const int K) {
    // check
    CHECK_INPUT(grad_pFeat);
    CHECK_INPUT(spIndx);
    // backward
    return spFeatSmear3d_cuda_backward(grad_pFeat, spIndx, K);
}

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spFeatSmear3d_forward, "spFeatSmear3d forward (CUDA)");
  m.def("backward", &spFeatSmear3d_backward, "spFeatSmear3d backward (CUDA)");
}
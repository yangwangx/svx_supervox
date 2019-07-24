#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// img_spFeat = smear_cuda.forward(spFeat, spIndx)

torch::Tensor smear_cuda_forward(
    const torch::Tensor spFeat,
    const torch::Tensor spIndx);  // return img_spFeat;

torch::Tensor smear_forward(
    const torch::Tensor spFeat,
    const torch::Tensor spIndx) {
    // check
    CHECK_INPUT(spFeat);
    CHECK_INPUT(spIndx);
    // forward
    return smear_cuda_forward(spFeat, spIndx);
}

/* Backward */
//  grad_spFeat = smear_cuda.backward(grad_img_spFeat, spIndx, K)

torch::Tensor smear_cuda_backward(
    const torch::Tensor grad_img_spFeat,
    const torch::Tensor spIndx,
    const int K);  // return grad_spFeat;

torch::Tensor smear_backward(
    const torch::Tensor grad_img_spFeat,
    const torch::Tensor spIndx,
    const int K) {
    // check
    CHECK_INPUT(grad_img_spFeat);
    CHECK_INPUT(spIndx);
    // backward
    return smear_cuda_backward(grad_img_spFeat, spIndx, K);
}

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smear_forward, "smear forward (CUDA)");
  m.def("backward", &smear_backward, "smear backward (CUDA)");
}
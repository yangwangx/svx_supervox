#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// absIndx = relToAbsIndex3d_cuda.forward(relIndx, init_spIndx, Kh, Kw)

torch::Tensor relToAbsIndex3d_cuda_forward(
    const torch::Tensor relIndx,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw);  // return absIndx;

torch::Tensor relToAbsIndex3d_forward(
    const torch::Tensor relIndx,
    const torch::Tensor init_spIndx,
    const int Kl,
    const int Kh,
    const int Kw) {
    // check
    CHECK_INPUT(relIndx);
    CHECK_INPUT(init_spIndx);
    // forward
    return relToAbsIndex3d_cuda_forward(relIndx, init_spIndx, Kl, Kh, Kw);
}

/* Backward */

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relToAbsIndex3d_forward, "relToAbsIndex3d forward (CUDA)");
}
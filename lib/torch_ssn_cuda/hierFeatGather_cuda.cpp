#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// hierFeat, hierSize = hierFeatGather_cuda.forward(spFeat, spSize, assign, K)

std::vector<torch::Tensor> hierFeatGather_cuda_forward(
    const torch::Tensor spFeat,
    const torch::Tensor spSize,
    const torch::Tensor assign,
    const int hierK);  // return {hierFeat, hierSize};

std::vector<torch::Tensor> hierFeatGather_forward(
    const torch::Tensor spFeat,
    const torch::Tensor spSize,
    const torch::Tensor assign,
    const int hierK) {
    // check
    CHECK_INPUT(spFeat);
    CHECK_INPUT(spSize);
    CHECK_INPUT(assign);
    // forward
    return hierFeatGather_cuda_forward(spFeat, spSize, assign, hierK);
}

/* Backward */
// nothing

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hierFeatGather_forward, "hierFeatGather forward (CUDA)");
}

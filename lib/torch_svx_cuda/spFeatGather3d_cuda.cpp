#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Forward */
// spFeat, spSize = spFeatGather3d_cuda.forward(pFeat, init_spIndx, K, ignore_idx_value, ignore_feature_value)

std::vector<torch::Tensor> spFeatGather3d_cuda_forward(
    const torch::Tensor pFeat,
    const torch::Tensor init_spIndx,
    const int K,
    const int ignore_idx_value,
    const int ignore_feature_value);  // return {spFeat, spSize};

std::vector<torch::Tensor> spFeatGather3d_forward(
    const torch::Tensor pFeat,
    const torch::Tensor init_spIndx,
    const int K,
    const int ignore_idx_value,
    const int ignore_feature_value) {
    // check
    CHECK_INPUT(pFeat);
    CHECK_INPUT(init_spIndx);
    // forward
    return spFeatGather3d_cuda_forward(pFeat, init_spIndx, K, ignore_idx_value, ignore_feature_value);
}

/* Backward */
// nothing

/* Python Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spFeatGather3d_forward, "spFeatGather3d forward (CUDA)");
}

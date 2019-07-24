from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_supervoxel_cuda',
    ext_modules=[
        CUDAExtension('spFeat_3d_cuda', [  # no backward
            'spFeat_cuda.cpp',
            'spFeat_cuda_kernel.cu',
        ]),
        CUDAExtension('spFeatUpdate_3d_cuda', [
            'spFeatUpdate_cuda.cpp',
            'spFeatUpdate_cuda_kernel.cu',
        ]),
        CUDAExtension('psp_sqdist_3d_cuda', [
            'psp_sqdist_cuda.cpp',
            'psp_sqdist_cuda_kernel.cu',
        ]),
        CUDAExtension('smear_3d_cuda', [
            'smear_cuda.cpp',
            'smear_cuda_kernel.cu',
        ]),
        CUDAExtension('relToAbsIndex_3d_cuda', [  # no backward
            'relToAbsIndex_cuda.cpp',
            'relToAbsIndex_cuda_kernel.cu',
        ]),
        CUDAExtension('hierFeat_cuda', [  # no backward
            'hierFeat_cuda.cpp',
            'hierFeat_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

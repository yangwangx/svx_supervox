from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_ssn_cuda',
    ext_modules=[
        CUDAExtension('spFeatGather2d_cuda', [  # no backward
            'spFeatGather2d_cuda.cpp',
            'spFeatGather2d_cuda_kernel.cu',
        ]),
        CUDAExtension('spFeatUpdate2d_cuda', [
            'spFeatUpdate2d_cuda.cpp',
            'spFeatUpdate2d_cuda_kernel.cu',
        ]),
        CUDAExtension('pspDist2d_cuda', [
            'pspDist2d_cuda.cpp',
            'pspDist2d_cuda_kernel.cu',
        ]),
        CUDAExtension('spFeatSmear2d_cuda', [
            'spFeatSmear2d_cuda.cpp',
            'spFeatSmear2d_cuda_kernel.cu',
        ]),
        CUDAExtension('relToAbsIndex2d_cuda', [  # no backward
            'relToAbsIndex2d_cuda.cpp',
            'relToAbsIndex2d_cuda_kernel.cu',
        ]),
        CUDAExtension('hierFeatGather_cuda', [  # no backward
            'hierFeatGather_cuda.cpp',
            'hierFeatGather_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


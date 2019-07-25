from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_svx_cuda',
    ext_modules=[
        CUDAExtension('spFeatGather3d_cuda', [  # no backward
            'spFeatGather3d_cuda.cpp',
            'spFeatGather3d_cuda_kernel.cu',
        ]),
        CUDAExtension('spFeatUpdate3d_cuda', [
            'spFeatUpdate3d_cuda.cpp',
            'spFeatUpdate3d_cuda_kernel.cu',
        ]),
        CUDAExtension('pspDist3d_cuda', [
            'pspDist3d_cuda.cpp',
            'pspDist3d_cuda_kernel.cu',
        ]),
        CUDAExtension('spFeatSmear3d_cuda', [
            'spFeatSmear3d_cuda.cpp',
            'spFeatSmear3d_cuda_kernel.cu',
        ]),
        CUDAExtension('relToAbsIndex3d_cuda', [  # no backward
            'relToAbsIndex3d_cuda.cpp',
            'relToAbsIndex3d_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


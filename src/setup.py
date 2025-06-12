from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='stegano_cuda',
    ext_modules=[
        CUDAExtension('stegano_cuda', [
            'cuda_ops.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

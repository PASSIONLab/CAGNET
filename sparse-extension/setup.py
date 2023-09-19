from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_coo_tensor_cpp',
      ext_modules=[cpp_extension.CUDAExtension('sparse_coo_tensor_cpp', ['sparse_coo_tensor.cu'],
                                                    extra_compile_args=["-O2", "-I/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.7/../../math_libs/include/", "-I/global/u1/a/alokt/nsparse/cuda-c/inc", "-I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/../../../22.7/examples/OpenACC/SDK/include/", "-lm -arch=sm_80 -lcusparse"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_coo_tensor_cpp',
      ext_modules=[cpp_extension.CUDAExtension('sparse_coo_tensor_cpp', ['sparse_coo_tensor.cu'],
                                                    extra_compile_args=["-lcusparse"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

setup(name='sparse_coo_tensor_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_coo_tensor_cpp', ['sparse_coo_tensor.cpp'],
                                        include_dirs=[os.path.join(os.environ["CUDA_HOME"], "include")],
                                        library_dirs=[os.path.join(os.environ["CUDA_HOME"], "lib64")],
                                        extra_compile_args=["-lcusparse"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

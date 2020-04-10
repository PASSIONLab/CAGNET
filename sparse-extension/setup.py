from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_coo_tensor_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_coo_tensor_cpp', ['sparse_coo_tensor.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

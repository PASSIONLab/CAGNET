#include <ATen/ATen.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include "cusparse.h"

#include <pybind11/pybind11.h>

#include <THC/THCGeneral.hpp>

#include <torch/extension.h>

namespace py = pybind11;

using namespace at::sparse;

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

at::Tensor expand_values_if_needed(const at::Tensor& values) {
    // expand
    if (values.dim() == 0) {
        // Mimic Numpy behavior here and treat it as a 1D tensor
        return values.expand({1});
    } else {
        return values;
    }
}

at::Tensor sparse_coo_tensor_gpu(const at::Tensor& indices, 
                                    const at::Tensor& values_, 
                                    at::ArrayRef<int64_t> size) {

    at::Tensor values = expand_values_if_needed(values_); 

    int64_t sparse_dim = indices.size(0);
    int64_t dense_dim = values.dim() - 1;

    return at::_sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, size, indices, values, values.options().layout(at::kSparse));
}

template<typename T>
void printCusparseDnMat(int64_t rows, int64_t cols, int64_t ld, T *values_dev) {
  T* values_host = new T[rows*cols];
  cudaMemcpy(values_host, values_dev, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t row = 0; row < rows; row++) {
    for (int64_t col = 0; col < cols; col++) {
      // Cusparse dense matrices are stored in column-major order
      std::cout << values_host[col*rows+row] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < rows*cols; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
}

template<typename T>
void printCusparseSpMat(int32_t rows, int32_t cols, int32_t nnz, int32_t *row_indices_dev,
                            int32_t *col_indices_dev, T *values_dev) {
  T* values_host = new T[nnz];
  int32_t* row_indices_host = new int32_t[nnz];
  int32_t* col_indices_host = new int32_t[nnz];
  cudaMemcpy(values_host, values_dev, nnz*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_indices_host, row_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_indices_host, col_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < nnz; i++) {
    std::cout << "(" << row_indices_host[i]
      << ", " << col_indices_host[i]
      << "): " << values_host[i] << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  row_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << row_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  col_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << col_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
  delete [] row_indices_host;
  delete [] col_indices_host;
}

at::Tensor spmm_gpu(const at::Tensor& A_rowindices, 
                        const at::Tensor& A_colindices,
                        const at::Tensor& A_values, 
                        int32_t n,
                        int32_t m,
                        at::Tensor& B) {

    // cusparseHandle_t handle;
    // CHECK_CUSPARSE(cusparseCreate(&handle));
    auto state = at::globalContext().lazyInitCUDA();
    auto handle = THCState_getCurrentSparseHandle(state);

    // Impl1 -- coo2csr + csrmm2
    int nnz = A_values.size(0);

    clock_t start, stop;
    
    int32_t *d_a_csrrows;
    cudaMalloc(&d_a_csrrows, (n + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        A_rowindices.data<int>(), 
                                        nnz, 
                                        n, 
                                        d_a_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));

    double alpha = 1;
    double beta = 0;
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    auto C = torch::ones({n, B.size(1)}, torch::dtype(torch::kDouble).device(torch::kCUDA));
    CHECK_CUSPARSE(cusparseDcsrmm2(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    n,
                                    B.size(1),
                                    m,
                                    nnz,
                                    &alpha,
                                    descrA,
                                    A_values.data<double>(),
                                    d_a_csrrows,
                                    A_colindices.data<int>(),
                                    B.data<double>(),
                                    B.size(1),
                                    &beta,
                                    C.data<double>(),
                                    n)); 

    // Impl1.5 -- coo2csr + cusparseSpMM
    // cusparseSpMatDescr_t matA;
    // cusparseDnMatDescr_t matB;
    // cusparseDnMatDescr_t matC;
    // CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, m, nnz,
    //                                   d_a_csrrows,
    //                                   A_colindices.data<int>(),
    //                                   A_values.data<double>(),
    //                                   CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_BASE_ZERO,
    //                                   CUDA_R_64F));

    // CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B.size(1), B.size(0), B.size(1),
    //                          B.data<double>(),
    //                          CUDA_R_64F,
    //                          CUSPARSE_ORDER_COL));

    // // Construct C
    // auto C = torch::ones({n, B.size(1)}, torch::dtype(torch::kDouble).device(torch::kCUDA));
    // CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C.size(0), C.size(1), C.size(0),
    //                         C.data<double>(),
    //                         CUDA_R_64F,
    //                         CUSPARSE_ORDER_COL));

    // // cusparseSpMM_bufferSize to get buffer size fro spmm call
    // size_t buffer_size;
    // double alpha = 1;
    // double beta = 0;
    // CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, 
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                 CUSPARSE_OPERATION_TRANSPOSE, // torch is row-m, cusparse col-m
    //                                 (void*)&alpha, matA, matB, 
    //                                 (void*)&beta, matC,
    //                                 CUDA_R_64F,
    //                                 CUSPARSE_CSRMM_ALG1,
    //                                 &buffer_size));
    // // allocate buffer
    // void *buffer = nullptr;
    // cudaMalloc(&buffer, buffer_size);

    // // cusparseSpMM with A, B, C and buffer
    // CHECK_CUSPARSE(cusparseSpMM(handle,
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                 CUSPARSE_OPERATION_TRANSPOSE,
    //                                 (void*)&alpha, matA, matB, 
    //                                 (void*)&beta, matC,
    //                                 CUDA_R_64F,
    //                                 CUSPARSE_CSRMM_ALG1,
    //                                 buffer));

    // Impl2 -- cusparseSpMM on coo
    // cusparseSpMatDescr_t matA;
    // cusparseDnMatDescr_t matB;
    // cusparseDnMatDescr_t matC;
    // // Construct A
    // int nnz = A_values.size(0);
    // CHECK_CUSPARSE(cusparseCreateCoo(&matA, n, m, nnz, 
    //                                     A_rowindices.data<int>(), 
    //                                     A_colindices.data<int>(), 
    //                                     A_values.data<double>(), 
    //                                     CUSPARSE_INDEX_32I, 
    //                                     CUSPARSE_INDEX_BASE_ZERO, 
    //                                     CUDA_R_64F));

    // 
    // // Construct B
    // B = B.view({B.size(1), B.size(0)}).t();

    // CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B.size(0), B.size(1), B.size(0),
    //                         B.data<double>(),
    //                         CUDA_R_64F,
    //                         CUSPARSE_ORDER_COL));

    // // Construct C
    // auto C = torch::ones({n, B.size(1)}, torch::dtype(torch::kDouble).device(torch::kCUDA));
    // CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C.size(0), C.size(1), C.size(0),
    //                         C.data<double>(),
    //                         CUDA_R_64F,
    //                         CUSPARSE_ORDER_COL));

    // // cusparseSpMM_bufferSize to get buffer size fro spmm call
    // size_t buffer_size;
    // double alpha = 1;
    // double beta = 0;
    // CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, 
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE, // torch is row-m, cusparse col-m
    //                                 (void*)&alpha, matA, matB, 
    //                                 (void*)&beta, matC,
    //                                 CUDA_R_64F,
    //                                 CUSPARSE_COOMM_ALG1,
    //                                 &buffer_size));
    // // allocate buffer
    // void *buffer = nullptr;
    // cudaMalloc(&buffer, buffer_size);

    // // cusparseSpMM with A, B, C and buffer
    // CHECK_CUSPARSE(cusparseSpMM(handle,
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                 (void*)&alpha, matA, matB, 
    //                                 (void*)&beta, matC,
    //                                 CUDA_R_64F,
    //                                 CUSPARSE_COOMM_ALG1,
    //                                 buffer));

    return C.view({B.size(1), n}).t();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
}

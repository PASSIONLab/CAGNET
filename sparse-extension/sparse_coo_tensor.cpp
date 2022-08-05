#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
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

#define CHECK_ERROR(str) \
    {cudaDeviceSynchronize(); cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout);}}


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

// at::Tensor spmm_gpu(const at::Tensor& A_rowindices, 
void spmm_gpu(const at::Tensor& A_rowindices, 
                        const at::Tensor& A_colindices,
                        const at::Tensor& A_values, 
                        int32_t n,
                        int32_t m,
                        at::Tensor& B,
                        at::Tensor& C) {

    // cusparseHandle_t handle;
    // CHECK_CUSPARSE(cusparseCreate(&handle));
    auto state = at::globalContext().lazyInitCUDA();
    // auto handle = THCState_getCurrentSparseHandle(state);
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // Impl1 -- coo2csr + csrmm2
    int nnz = A_values.size(0);

    clock_t start, stop;
    
    int32_t *d_a_csrrows;
    
    // int devid_old = 0;
    // cudaGetDevice(&devid_old);
    // cudaSetDevice(devid);

    cudaMalloc(&d_a_csrrows, (n + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        A_rowindices.data<int>(), 
                                        nnz, 
                                        n, 
                                        d_a_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));

    int32_t b_row = B.size(0);
    int32_t b_col = B.size(1);
    int32_t c_row = C.size(0);
    int32_t c_col = C.size(1);

    float alpha = 1;
    float beta = 1;
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
					  n, 		// rows
					  // b_col, 	// cols
					  m, 	// cols
					  nnz, 		// nnz
					  d_a_csrrows, 	// csrRowOffsets
					  A_colindices.data<int>(), // csrColInd
					  A_values.data<float>(),   // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType

    // Row-major to column-major
    B.t_();
    B.set_data(B.contiguous());
    B.set_data(B.view({b_row, b_col}));

    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, 
                                            // b_col, // rows
                                            b_row, // rows
                                            // b_row, // cols
                                            b_col, // cols
                                            // b_col, // ld
                                            b_row, // ld
                                            B.data<float>(), // values
                                            CUDA_R_32F,      // valueType
                                            CUSPARSE_ORDER_COL)); // order
        
    // Row-major to column-major
    C.t_();
    C.set_data(C.contiguous());
    C.set_data(C.view({c_row, c_col}));

    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, 
                                            n, // rows
                                            B.size(1), // cols
                                            // n, // ld
                                            n, // ld
                                            C.data<float>(), // values
                                            CUDA_R_32F,      // valueType
                                            CUSPARSE_ORDER_COL)); // order
	
    size_t bufferSize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, // handle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
                                                // CUSPARSE_OPERATION_TRANSPOSE,       // opB
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,       // opB
                                                &alpha,                             // alpha
                                                matA,                               // matA
                                                matB,                               // matB
                                                &beta,                              // beta
                                                matC,                               // matC
                                                CUDA_R_32F,                         // computeType
                                                CUSPARSE_CSRMM_ALG1,                // alg
                                                &bufferSize));                      // bufferSize


    void* d_buffer = NULL;
    CHECK_ERROR(cudaMalloc(&d_buffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(handle, // handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
                                    // CUSPARSE_OPERATION_TRANSPOSE,       // opB
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,       // opB
                                    &alpha,                             // alpha
                                    matA,                               // matA
                                    matB,                               // matB
                                    &beta,                              // beta
                                    matC,                               // matC
                                    CUDA_R_32F,                         // computeType
                                    CUSPARSE_CSRMM_ALG1,                // alg
                                    d_buffer));                         // buffer


    CHECK_ERROR(cudaFree(d_a_csrrows));
    CHECK_ERROR(cudaFree(d_buffer));

    // Column-major to row-major
    C.set_data(C.view({c_col, c_row}));
    C.t_();

    // Column-major to row-major
    B.set_data(B.view({b_col, b_row}));
    B.t_();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
}

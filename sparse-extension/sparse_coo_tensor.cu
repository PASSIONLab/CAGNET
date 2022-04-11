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
					  m, 	        // cols
					  nnz, 		// nnz
					  d_a_csrrows, 	// csrRowOffsets
					  A_colindices.data<int>(), // csrColInd
					  A_values.data<float>(),   // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType

    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, 
                                            b_col, // rows
                                            b_row, // cols
                                            b_col, // ld
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
                                            b_col, // cols
                                            n, // ld
                                            C.data<float>(), // values
                                            CUDA_R_32F,      // valueType
                                            CUSPARSE_ORDER_COL)); // order
	
    size_t bufferSize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, // handle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
                                                CUSPARSE_OPERATION_TRANSPOSE,   // opB
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
                                    CUSPARSE_OPERATION_TRANSPOSE,   // opB
                                    &alpha,                             // alpha
                                    matA,                               // matA
                                    matB,                               // matB
                                    &beta,                              // beta
                                    matC,                               // matC
                                    CUDA_R_32F,                         // computeType
                                    CUSPARSE_CSRMM_ALG1,                // alg
                                    d_buffer));                         // buffer


    cudaFree(d_a_csrrows);
    cudaFree(d_buffer);

    // Column-major to row-major
    C.set_data(C.view({c_col, c_row}));
    C.t_();
}

// __global__ void DownSample(long *q_values, long *q_rows, int *overflow, int nnz) {
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
// 
//     for (int i = id; i < nnz; i += stride) {
//         long row = q_rows[i];
//         if (q_values[i] == 1 && ((int) atomicSub((unsigned int *)&overflow[row], 1)) > 0) {
//             q_values[i] = 0;
//         }
//     }
// }

// void downsample_gpu(const at::Tensor& q_values, 
//                         const at::Tensor& q_rows,
//                         const at::Tensor& overflow,
//                         int nnz) {
// 
// 
//     int BLOCK_SIZE = 256;
//     int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
//     BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);
// 
//     DownSample<<<BLOCK_COUNT, BLOCK_SIZE>>>(q_values.data<long>(), q_rows.data<long>(), 
//                                                                 overflow.data<int>(), nnz);
//     CHECK_ERROR("downsampling error")
// }

__global__ void DownSample(long *h_counts, long *h_rows, long *ps_h_rows, long *hev_indices, int *overflow, 
                                int nnz) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < nnz; i += stride) {
        if (i - ps_h_rows[h_rows[i]] < overflow[h_rows[i]]) {
            h_counts[hev_indices[i]] = 0;
        } else if (h_rows[i] == 209) {
            printf("i: %d hev_indices[i]: %ld h_counts[hev_indices[i]]: %ld\n", i, hev_indices[i], h_counts[hev_indices[i]]);
        }
    }
}

void downsample_gpu(const at::Tensor& h_counts, 
                        const at::Tensor& h_rows,
                        const at::Tensor& ps_h_rows,
                        const at::Tensor& hev_indices,
                        const at::Tensor& overflow,
                        int nnz) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    DownSample<<<BLOCK_COUNT, BLOCK_SIZE>>>(h_counts.data<long>(), 
                                                h_rows.data<long>(), 
                                                ps_h_rows.data<long>(), 
                                                hev_indices.data<long>(), 
                                                overflow.data<int>(), 
                                                nnz);
    CHECK_ERROR("downsampling error")
}

__global__ void ComputeDarts(float *dartx_values, float *darty_values, long *neighbor_sizes, 
                                long *psum_neighbor_sizes, float *pmax, int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int dart_count = n_darts * mb_count;
    for (int i = id; i < dart_count; i += stride) {
        long row = i / n_darts;
        dartx_values[i] *= neighbor_sizes[row];
        dartx_values[i] += psum_neighbor_sizes[row];
        darty_values[i] *= pmax[row];
    }
}

void compute_darts_gpu(const at::Tensor& dartx_values, 
                        const at::Tensor& darty_values,
                        const at::Tensor& neighbor_sizes,
                        const at::Tensor& psum_neighbor_sizes,
                        const at::Tensor& pmax,
                        int n_darts,
                        int mb_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((n_darts * mb_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ComputeDarts<<<BLOCK_COUNT, BLOCK_SIZE>>>(dartx_values.data<float>(), 
                                                darty_values.data<float>(), 
                                                neighbor_sizes.data<long>(), 
                                                psum_neighbor_sizes.data<long>(), 
                                                pmax.data<float>(), 
                                                n_darts,
                                                mb_count);
    CHECK_ERROR("dart computation error")
}

__global__ void ThrowDarts(float *dartx_values, float *darty_values, float *p_values, 
                                long *h_values, int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int dart_count = n_darts * mb_count;
    for (int i = id; i < dart_count; i += stride) {
        long dartx_val = (long) dartx_values[i];
        if (darty_values[i] < p_values[dartx_val]) {
            atomicAdd((unsigned long long *)&h_values[dartx_val], 1L);
        }
    }
}

void throw_darts_gpu(const at::Tensor& dartx_values, 
                        const at::Tensor& darty_values,
                        const at::Tensor& p_values,
                        const at::Tensor& h_values,
                        int n_darts,
                        int mb_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((n_darts * mb_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ThrowDarts<<<BLOCK_COUNT, BLOCK_SIZE>>>(dartx_values.data<float>(), 
                                                darty_values.data<float>(), 
                                                p_values.data<float>(), 
                                                h_values.data<long>(), 
                                                n_darts,
                                                mb_count);
    CHECK_ERROR("dart throwing error")
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
    m.def("downsample_gpu", &downsample_gpu, "Downsampling for LADIES sampling algorithm");
    m.def("compute_darts_gpu", &compute_darts_gpu, "Compute dart values for LADIES sampling algorithm");
    m.def("throw_darts_gpu", &throw_darts_gpu, "Throw darts in LADIES sampling algorithm");
}

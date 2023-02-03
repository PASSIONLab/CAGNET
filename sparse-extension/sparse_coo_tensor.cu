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

// #include <THC/THCGeneral.hpp>

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

__device__ int binary_searchf(double *arr, double val, int imin, int imax) {
    
    int ans = 0;
    while (imax >= imin) {
        int imid = (imin + imax) / 2;

        if (arr[imid] < val) {
            imin = imid + 1;
        } else {
            ans = imid;
            imax = imid - 1;
        }
    }

    return ans;
}

__device__ int binary_searchfpf(double *arr, double val, int imin, int imax) {

    int ans = 0;
    while (imax >= imin) {
        int imid = (imin + imax) / 2;
        printf("imid: %d val: %f arr[imid]: %f\n", imid, val, arr[imid]);

        if (arr[imid] < val) {
            imin = imid + 1;
        } else {
            ans = imid;
            imax = imid - 1;
        }
    }

    return ans;
}

__device__ long binary_searchl(long *arr, long val, long imin, long imax) {
    
    long ans = -1;
    while (imax >= imin) {
        long imid = (imin + imax) / 2;

        if (arr[imid] <= val) {
            imin = imid + 1;
        } else if (arr[imid] > val) {
            ans = imid;
            imax = imid - 1;
        }
    }

    return ans;
}

// Binary search that returns exact location, not insertion point (for COO row selection)
__device__ long binary_search_rowselect(long *arr, long val, long imin, long imax) {
    while (imax >= imin) {
        long imid = (imin + imax) / 2;

        if (arr[imid] < val) {
            imin = imid + 1;
        } else if (arr[imid] > val) {
            imax = imid - 1;
        } else {
            return imid;
        }
    }

    return imin + 1;
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

// at::Tensor spmm_gpu(const at::Tensor& A_rowindices, 
void spmm_gpu(const at::Tensor& A_rowindices, 
                        const at::Tensor& A_colindices,
                        const at::Tensor& A_values, 
                        int32_t n,
                        int32_t m,
                        at::Tensor& B,
                        at::Tensor& C) {

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
    // CHECK_ERROR(cudaMalloc(&d_buffer, bufferSize));
    cudaMalloc(&d_buffer, bufferSize);

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
std::vector<at::Tensor> coogeam_gpu(
        const at::Tensor& A_rowindices, 
        const at::Tensor& A_colindices,
        const at::Tensor& A_values, 
        const at::Tensor& B_rowindices, 
        const at::Tensor& B_colindices,
        const at::Tensor& B_values,
        int32_t m,
        int32_t n) {


    // sum matrix A (m x n) and B (m x n)
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    int32_t *d_a_csrrows = NULL;
    int32_t *d_b_csrrows = NULL;

    int32_t A_nnz = A_values.size(0);
    int32_t B_nnz = B_values.size(0);


    // Construct CSR offsets array for A and B
    cudaMalloc(&d_a_csrrows, (m + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        A_rowindices.data<int>(), 
                                        A_nnz, 
                                        m,
                                        d_a_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));

    cudaMalloc(&d_b_csrrows, (m + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        B_rowindices.data<int>(), 
                                        B_nnz, 
                                        m, 
                                        d_b_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));
    
    double alpha = 1;
    double beta = 1;

    // Construct CSR matrix structs for A, B, and C (empty CSR)
    cusparseMatDescr_t matA;
    cusparseCreateMatDescr(&matA);

    cusparseMatDescr_t matB;
    cusparseCreateMatDescr(&matB);

    int32_t *d_c_csrrows = NULL;
    int32_t *d_c_columns = NULL;
    double *d_c_values = NULL;

    cudaMalloc(&d_c_csrrows, (m + 1) * sizeof(int32_t));
    cusparseMatDescr_t matC;
    cusparseCreateMatDescr(&matC);

    char *buffer = NULL;
    size_t bufferSize;
    CHECK_CUSPARSE(cusparseDcsrgeam2_bufferSizeExt(handle,    // handle,
                                    m,                        //  m,
                                    n,                        //  n,
                                    &alpha,                   //  alpha,
                                    matA,                     //  descrA,
                                    A_nnz,                    //  nnzA,
                                    A_values.data<double>(),  //  csrSortedValA,
                                    d_a_csrrows,              //  csrSortedRowPtrA,
                                    A_colindices.data<int>(), //  csrSortedColIndA,
                                    &beta,                    //  beta,
                                    matB,                     //  descrB,
                                    B_nnz,                    //  nnzB,
                                    B_values.data<double>(),  //  csrSortedValB,
                                    d_b_csrrows,              //  csrSortedRowPtrB,
                                    B_colindices.data<int>(), //  csrSortedColIndB,
                                    matC,                     //  descrC,
                                    d_c_values,               //  csrSortedValC,
                                    d_c_csrrows,              //  csrSortedRowPtrC,
                                    d_c_columns,              //  csrSortedColIndC,
                                    &bufferSize));            //  pBufferSizeInBytes)

    cudaMalloc(&buffer, bufferSize * sizeof(char));

    int32_t C_nnz;
    int32_t *C_nnz_ptr = &C_nnz;
    // std::cout << "m: " << m << std::endl;
    // std::cout << "n: " << n << std::endl;
    // std::cout << "A_nnz: " << A_nnz << std::endl;
    // std::cout << "B_nnz: " << B_nnz << std::endl;
    // std::cout << "d_a_csrrows: " << d_a_csrrows << std::endl;
    // std::cout << "d_b_csrrows: " << d_b_csrrows << std::endl;
    // std::cout << "d_c_csrrows: " << d_c_csrrows << std::endl;
    // std::cout << "A_colindices.size(0): " << A_colindices.size(0) << std::endl;
    // std::cout << "B_colindices.size(0): " << B_colindices.size(0) << std::endl;
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle,                         // handle,
                                             m,                         // m,
                                             n,                         // n,
                                             matA,                      // descrA,
                                             A_nnz,                     // nnzA,
                                             d_a_csrrows,               // csrSortedRowPtrA,
                                             A_colindices.data<int>(),  // csrSortedColIndA,
                                             matB,                      // descrB,
                                             B_nnz,                     // nnzB,
                                             d_b_csrrows,               // csrSortedRowPtrB,
                                             B_colindices.data<int>(),  // csrSortedColIndB,
                                             matC,                      // descrC,
                                             d_c_csrrows,               // csrSortedRowPtrC,
                                             C_nnz_ptr,                 // nnzTotalDevHostPtr,
                                             buffer));                 // workspace
    CHECK_ERROR("after 2nnz");
    if (NULL != C_nnz_ptr) {
        C_nnz = *C_nnz_ptr;
    } else {
        int32_t baseC;
        cudaMemcpy(&C_nnz, d_c_csrrows + m, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, d_c_csrrows, sizeof(int32_t), cudaMemcpyDeviceToHost);
        C_nnz -= baseC;
    }

    cudaMalloc(&d_c_columns, C_nnz * sizeof(int32_t));
    cudaMalloc(&d_c_values, C_nnz * sizeof(double));

    CHECK_CUSPARSE(cusparseDcsrgeam2(handle,                            // handle,
                                          m,                            // m,
                                          n,                            // n,
                                          &alpha,                       // alpha,
                                          matA,                         // descrA,
                                          A_nnz,                        // nnzA,
                                          A_values.data<double>(),      // csrSortedValA,
                                          d_a_csrrows,                  // csrSortedRowPtrA,
                                          A_colindices.data<int>(),     // csrSortedColIndA,
                                          &beta,                        // beta,
                                          matB,                         // descrB,
                                          B_nnz,                        // nnzB,
                                          B_values.data<double>(),      // csrSortedValB,
                                          d_b_csrrows,                  // csrSortedRowPtrB,
                                          B_colindices.data<int>(),     // csrSortedColIndB,
                                          matC,                         // descrC,
                                          d_c_values,                   // csrSortedValC,
                                          d_c_csrrows,                  // csrSortedRowPtrC,
                                          d_c_columns,                  // csrSortedColIndC,
                                          buffer));                     // pBuffer)

    auto rows_dim = torch::IntArrayRef{m + 1};
    auto nnz_dim = torch::IntArrayRef{C_nnz};

    auto c_csrOffsets = torch::from_blob(d_c_csrrows, rows_dim, torch::TensorOptions()
                                                                    .dtype(torch::kInt32)
                                                                    .device(torch::kCUDA));
    auto c_columns = torch::from_blob(d_c_columns, nnz_dim, torch::TensorOptions()
                                                                    .dtype(torch::kInt32)
                                                                    .device(torch::kCUDA));
    auto c_values = torch::from_blob(d_c_values, nnz_dim, torch::TensorOptions()
                                                                    .dtype(torch::kFloat64)
                                                                    .device(torch::kCUDA));
    cudaFree(buffer);

    CHECK_CUSPARSE(cusparseDestroyMatDescr(matA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(matB));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(matC));

    return {c_csrOffsets, c_columns, c_values};
}

std::vector<at::Tensor> spgemm_gpu(
        const at::Tensor& A_rowindices, 
        const at::Tensor& A_colindices,
        const at::Tensor& A_values, 
        const at::Tensor& B_rowindices, 
        const at::Tensor& B_colindices,
        const at::Tensor& B_values,
        int32_t n,
        int32_t k,
        int32_t m) {

    
    // SpGEMM between A (n x k)  and B (k x m)
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    int32_t *d_a_csrrows = NULL;
    int32_t *d_b_csrrows = NULL;

    int32_t A_nnz = A_values.size(0);
    int32_t B_nnz = B_values.size(0);


    // Construct CSR offsets array for A and B
    cudaMalloc(&d_a_csrrows, (n + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        A_rowindices.data<int>(), 
                                        A_nnz, 
                                        n, 
                                        d_a_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));
    cudaMalloc(&d_b_csrrows, (k + 1) * sizeof(int32_t));
    
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        B_rowindices.data<int>(), 
                                        B_nnz, 
                                        k, 
                                        d_b_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));
    
    float alpha = 1;
    float beta = 0;

    // Construct CSR matrix structs for A, B, and C (empty CSR)
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
					  n, 		// rows
					  k, 	        // cols
					  A_nnz, 	// nnz
					  d_a_csrrows, 	// csrRowOffsets
					  A_colindices.data<int>(), // csrColInd
					  A_values.data<float>(),  // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType

    cusparseSpMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateCsr(&matB,
					  k, 		// rows
					  m, 	        // cols
					  B_nnz, 	// nnz
					  d_b_csrrows, 	// csrRowOffsets
					  B_colindices.data<int>(), // csrColInd
					  B_values.data<float>(),  // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType

    cusparseSpMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matC,
					  n, 		// rows
					  m, 	        // cols
					  0, 	        // nnz
					  NULL, 	// csrRowOffsets
					  NULL,         // csrColInd
					  NULL,         // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType
    
    cusparseSpGEMMDescr_t spgemmDescr;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDescr));

    void *dBuffer1 = NULL;
    void *dBuffer2 = NULL;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,            // handle
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                  &alpha,                           // alpha
                                  matA,                             // matA
                                  matB,                             // matB
                                  &beta,                            // beta
                                  matC,                             // matC
                                  CUDA_R_32F,                       // computeType
                                  CUSPARSE_SPGEMM_DEFAULT,          // alg
                                  spgemmDescr,                      // spgemmDescr
                                  &bufferSize1,                     // bufferSize1
                                  NULL));                           // externalBuffer1

    //CHECK_ERROR(cudaMalloc(&dBuffer1, bufferSize1));
    cudaMalloc(&dBuffer1, bufferSize1);

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,            // handle
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                  &alpha,                           // alpha
                                  matA,                             // matA
                                  matB,                             // matB
                                  &beta,                            // beta
                                  matC,                             // matC
                                  CUDA_R_32F,                       // computeType
                                  CUSPARSE_SPGEMM_DEFAULT,          // alg
                                  spgemmDescr,                      // spgemmDescr
                                  &bufferSize1,                     // bufferSize1
                                  dBuffer1));                       // externalBuffer1

    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
			  	CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
			   	&alpha,				  // alpha
			   	matA, 				  // matA
			   	matB,				  // matB
			   	&beta,				  // beta
			   	matC,				  // matC
			  	CUDA_R_32F,                       // computeType
                                CUSPARSE_SPGEMM_DEFAULT,          // alg
                                spgemmDescr,                      // spgemmDescr
			   	&bufferSize2,			  // bufferSize2
			   	NULL));				  // externalBuffer2

    cudaMalloc(&dBuffer2, bufferSize2);

    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
			  	CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
			   	&alpha,				  // alpha
			   	matA, 				  // matA
			   	matB,				  // matB
			   	&beta,				  // beta
			   	matC,				  // matC
			  	CUDA_R_32F,                       // computeType
                                CUSPARSE_SPGEMM_DEFAULT,          // alg
                                spgemmDescr,                      // spgemmDescr
			   	&bufferSize2,			  // bufferSize2
			   	dBuffer2));			  // externalBuffer2

    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1));
    
    int32_t *dC_csrOffsets;
    int32_t *dC_columns;
    float *dC_values;
    cudaMalloc(&dC_csrOffsets, (n + 1) * sizeof(int32_t));
    cudaMalloc(&dC_columns, C_nnz1 * sizeof(int32_t));
    cudaMalloc(&dC_values,  C_nnz1 * sizeof(float));

    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                            CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                            &alpha,                           // alpha
                            matA,                             // matA
                            matB,                             // matB
                            &beta,                            // beta
                            matC,                             // matC
                            CUDA_R_32F,                       // computeType
                            CUSPARSE_SPGEMM_DEFAULT,          // alg
                            spgemmDescr));
    
    auto rows_dim = torch::IntArrayRef{n + 1};
    auto nnz_dim = torch::IntArrayRef{C_nnz1};

    auto c_csrOffsets = torch::from_blob(dC_csrOffsets, rows_dim, torch::TensorOptions()
                                                                    .dtype(torch::kInt32)
                                                                    .device(torch::kCUDA));
    auto c_columns = torch::from_blob(dC_columns, nnz_dim, torch::TensorOptions()
                                                                    .dtype(torch::kInt32)
                                                                    .device(torch::kCUDA));
    auto c_values = torch::from_blob(dC_values, nnz_dim, torch::TensorOptions()
                                                                    .dtype(torch::kFloat32)
                                                                    .device(torch::kCUDA));

    CHECK_ERROR("spgemm_gpu error");
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);

    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDescr));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));

    return {c_csrOffsets, c_columns, c_values};
}


__global__ void DownSample(long *h_counts, long *h_rows, long *ps_h_rows, long *hev_indices, int *overflow, 
                                int nnz) {

    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = id; i < nnz; i += stride) {
        if (i - ps_h_rows[h_rows[i]] < overflow[h_rows[i]]) {
            h_counts[hev_indices[i]] = 0;
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

__global__ void ComputeDarts1D(double *dart_values, double *p_rowsum, double *ps_p_rowsum, 
                                    int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int dart_count = n_darts * mb_count;
    for (int i = id; i < dart_count; i += stride) {
        int row = i / n_darts;
        // dart_values[i] += (double)row;
        dart_values[i] *= p_rowsum[row];
        dart_values[i] += ps_p_rowsum[row];
    }
}

void compute_darts1d_gpu(const at::Tensor& dart_values, const at::Tensor& p_rowsum, 
				const at::Tensor& ps_p_rowsum, int n_darts, int mb_count) {

    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((n_darts * mb_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ComputeDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<double>(), 
                                                p_rowsum.data<double>(),
                                                ps_p_rowsum.data<double>(),
                                                n_darts,
                                                mb_count);
    CHECK_ERROR("dart1d computation error")
}

__global__ void ComputeDartsSelect(double *dart_select, double *dart_hits_inv_sum, double *ps_dart_hits_inv_sum, 
                                long *ps_overflow, long mb_count, long total_overflow) {

    long      id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = id; i < total_overflow; i += stride) {
        long row = binary_searchl(ps_overflow, i, 0, mb_count);
        dart_select[i] *= dart_hits_inv_sum[row];
        dart_select[i] += ps_dart_hits_inv_sum[row];
    }
}

void compute_darts_select_gpu(const at::Tensor& dart_select, 
                                const at::Tensor& dart_hits_inv_sum,
                                const at::Tensor& ps_dart_hits_inv_sum,
                                const at::Tensor& ps_overflow,
                                long mb_count,
                                long total_overflow) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(total_overflow / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (total_overflow == 0) {
        return;
    }

    ComputeDartsSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_select.data<double>(), 
                                                        dart_hits_inv_sum.data<double>(), 
                                                        ps_dart_hits_inv_sum.data<double>(), 
                                                        ps_overflow.data<long>(), 
                                                        mb_count,
                                                        total_overflow);
    CHECK_ERROR("selection dart computation error")
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

__global__ void ThrowDarts1D(double *dart_values, double *ps_p_values, int *h_values, 
                                int dart_count, int nnz) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < dart_count; i += stride) {
        int vtx = binary_searchf(ps_p_values, dart_values[i], 0, nnz - 1);
        // if (vtx < 0 || vtx >= nnz) {
        //     printf("error i: %d vtx: %d nnz: %d\n", i, vtx, nnz);
        // } 
        atomicAdd(&h_values[vtx], 1);
    }
}

void throw_darts1d_gpu(const at::Tensor& dart_values, 
                            const at::Tensor& ps_p_values,
                            const at::Tensor& h_values,
                            int dart_count,
                            int nnz) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((dart_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ThrowDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<double>(), 
                                                ps_p_values.data<double>(), 
                                                h_values.data<int>(), 
                                                dart_count,
                                                nnz);
    CHECK_ERROR("dart throwing error")
}

__global__ void ThrowDartsSelect(double *dart_select, double *ps_dart_hits_inv, int *dart_hits_count, 
                                    int total_overflow, int nnz) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < total_overflow; i += stride) {
        int vtx = binary_searchf(ps_dart_hits_inv, dart_select[i], 0, nnz);
        atomicAnd(&dart_hits_count[vtx], 0);
    }
}

void throw_darts_select_gpu(const at::Tensor& dart_select, 
                                const at::Tensor& ps_dart_hits_inv,
                                const at::Tensor& dart_hits_count,
                                int total_overflow,
                                int nnz) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(total_overflow / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (total_overflow == 0) {
        return;
    }

    ThrowDartsSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_select.data<double>(), 
                                                    ps_dart_hits_inv.data<double>(), 
                                                    dart_hits_count.data<int>(), 
                                                    total_overflow,
                                                    nnz);
    CHECK_ERROR("selection dart throwing error")
}

__global__ void Normalize(double *output, double *input, long *index, int len) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < len; i += stride) {
        output[i] /= input[index[i]];
    }
}

void normalize_gpu(const at::Tensor& output, const at::Tensor& input, const at::Tensor& index, int len) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    Normalize<<<BLOCK_COUNT, BLOCK_SIZE>>>(output.data<double>(), input.data<double>(), index.data<long>(), len);
    CHECK_ERROR("normalize error")
}

__global__ void ShiftRowSelect(long *row_shift, long *row_select_rows, int rank, int size,
                                    int replication, int nnz, int batch_size, int node_count, int mb_count,
                                    int shift_size) { 

    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    int rank_c = rank / replication;
    long proc_row_chunk = rank_c * ((batch_size * mb_count) / (size / replication));
    
    for (int i = id; i < nnz; i += stride) {
        long mb_row = (row_select_rows[i] + proc_row_chunk) / shift_size;
        row_shift[i] += mb_row * node_count;
    }
}

void shift_rowselect_gpu(const at::Tensor& row_shift, const at::Tensor& row_select_rows,
                            int nnz, int rank, int size, int replication, int batch_size, int node_count, 
                            int mb_count, int shift_size) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ShiftRowSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(row_shift.data<long>(), 
                                                    row_select_rows.data<long>(), 
                                                    rank,
                                                    size,
                                                    replication,
                                                    nnz,
                                                    batch_size,
                                                    node_count,
                                                    mb_count,
                                                    shift_size);
    CHECK_ERROR("shift row select error")
}

__global__ void ShiftColSelect(long *col_shift, int nnz, int batch_size, int node_count) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < nnz; i += stride) {
        long mb_row = i / batch_size;
        col_shift[i] += mb_row * node_count;
    }
}

void shift_colselect_gpu(const at::Tensor& col_shift, int nnz, int batch_size, int node_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ShiftColSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(col_shift.data<long>(), nnz, batch_size, node_count);
    CHECK_ERROR("shift col select error")
}

__global__ void ScatterAddD(double *src, long *indices, double *values, int num_vals) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < num_vals; i += stride) {
        atomicAdd(&src[indices[i]], values[i]);
    } 
}

void scatterd_add_gpu(const at::Tensor& src, const at::Tensor& indices, const at::Tensor& values, int num_vals) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(num_vals / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ScatterAddD<<<BLOCK_COUNT, BLOCK_SIZE>>>(src.data<double>(), 
                                                indices.data<long>(), 
                                                values.data<double>(),
                                                num_vals);
    CHECK_ERROR("scatter add doubles error")
}

__global__ void ScatterAddI(int *src, long *indices, int *values, int num_vals) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < num_vals; i += stride) {
        atomicAdd(&src[indices[i]], values[i]);
    } 
}

void scatteri_add_gpu(const at::Tensor& src, const at::Tensor& indices, const at::Tensor& values, int num_vals) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(num_vals / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ScatterAddI<<<BLOCK_COUNT, BLOCK_SIZE>>>(src.data<int>(), 
                                                indices.data<long>(), 
                                                values.data<int>(),
                                                num_vals);
    CHECK_ERROR("scatter add ints error")
}

// // Per-process rowselect coo
// __global__ void RowSelectCoo(long *nnz_cols, long *row_ids, bool *mask, int nnz_col_count, int row_count) { 
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
// 
//     for (int i = id; i < row_count; i += stride) {
//         long idx = binary_search_rowselect(nnz_cols, row_ids[i], 0L, nnz_col_count);
//         if (nnz_cols[idx] == row_ids[i]) {
//             mask[i] = true;
//         }
//     } 
// }
// 
// void rowselect_coo_gpu(const at::Tensor& nnz_cols, const at::Tensor& rows, const at::Tensor& mask, 
//                             int nnz_col_count, int row_count) {
// 
// 
//     int BLOCK_SIZE = 256;
//     int BLOCK_COUNT = std::ceil(row_count / ((float) BLOCK_SIZE));
//     BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);
// 
//     if (nnz_col_count == 0 || row_count == 0) {
//         return;
//     }
// 
//     RowSelectCoo<<<BLOCK_COUNT, BLOCK_SIZE>>>(nnz_cols.data<long>(), 
//                                                 rows.data<long>(), 
//                                                 mask.data<bool>(), 
//                                                 nnz_col_count,
//                                                 row_count);
//     CHECK_ERROR("rowselect coo error")
// }

// Bulk rowselect coo
__global__ void RowSelectCoo(long **nnz_cols, long *row_ids, bool *mask, int *nnz_cols_counts, int row_count,
                                int proc_count) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int mask_len = row_count * proc_count;
    for (int i = id; i < mask_len; i += stride) {
        long proc = ((long) i) / row_count;
        long col = ((long) i) % row_count;
        if (nnz_cols_counts[proc] > 0) {
            long idx = binary_search_rowselect(nnz_cols[proc], row_ids[col], 0L, (long) nnz_cols_counts[proc]);
            if (nnz_cols[proc][idx] == row_ids[col]) {
                mask[i] = true;
            }
        }
    } 
}

void rowselect_coo_gpu(std::vector<at::Tensor> nnz_cols, const at::Tensor& rows, 
                            const at::Tensor& mask, const at::Tensor& nnz_cols_counts, 
                            int row_count, int proc_count) {


    int mask_len = row_count * proc_count;
    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(mask_len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (mask_len == 0) {
        return;
    }

    std::vector<long*> nnz_cols_hptr(nnz_cols.size());
    for (int i = 0; i < nnz_cols.size(); i++) {
        nnz_cols_hptr[i] = nnz_cols[i].data<long>();
    }

    long **nnz_cols_dptr;
    cudaMalloc(&nnz_cols_dptr, nnz_cols.size() * sizeof(long*));
    cudaMemcpy(nnz_cols_dptr, nnz_cols_hptr.data(), nnz_cols.size() * sizeof(long*), cudaMemcpyHostToDevice);

    RowSelectCoo<<<BLOCK_COUNT, BLOCK_SIZE>>>(nnz_cols_dptr, 
                                                rows.data<long>(), 
                                                mask.data<bool>(), 
                                                nnz_cols_counts.data<int>(),
                                                row_count,
                                                proc_count);
    CHECK_ERROR("rowselect coo error")
}

// Per-process rowselect csr
__global__ void RowSelectCsr(long *nnz_cols, long *row_offsets, bool *mask, int nnz_col_count) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < nnz_col_count; i += stride) {
        long vtx = nnz_cols[i];
        long degree = row_offsets[vtx + 1] - row_offsets[vtx];
        // might be able to use memset here
        for (int j = 0; j < degree; j++) {
            mask[row_offsets[vtx] + j] = true;
        }
    } 
}

void rowselect_csr_gpu(const at::Tensor& nnz_cols, const at::Tensor& row_offsets, const at::Tensor& mask, 
                            int nnz_col_count, int nnz_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz_col_count / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (nnz_col_count == 0 || nnz_count == 0) {
        return;
    }

    RowSelectCsr<<<BLOCK_COUNT, BLOCK_SIZE>>>(nnz_cols.data<long>(), 
                                                row_offsets.data<long>(), 
                                                mask.data<bool>(), 
                                                nnz_col_count);
    CHECK_ERROR("rowselect csr error")
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
    m.def("spgemm_gpu", &spgemm_gpu, "SpGEMM wrapper for cusparse");
    m.def("downsample_gpu", &downsample_gpu, "Downsampling for LADIES sampling algorithm");
    m.def("compute_darts_gpu", &compute_darts_gpu, "Compute dart values for LADIES sampling algorithm");
    m.def("throw_darts_gpu", &throw_darts_gpu, "Throw darts in LADIES sampling algorithm");
    m.def("compute_darts_select_gpu", &compute_darts_select_gpu, "Compute dart values for LADIES alg selection");
    m.def("throw_darts_select_gpu", &throw_darts_select_gpu, "Throw darts for LADIES alg selection");
    m.def("compute_darts1d_gpu", &compute_darts1d_gpu, "Compute 1D dart values for LADIES sampling algorithm");
    m.def("throw_darts1d_gpu", &throw_darts1d_gpu, "Throw 1D darts in LADIES sampling algorithm");
    m.def("normalize_gpu", &normalize_gpu, "Normalize values in an array based on a second array");
    m.def("shift_rowselect_gpu", &shift_rowselect_gpu, "Shift row selection output matrix col values");
    m.def("shift_colselect_gpu", &shift_colselect_gpu, "Shift col selection matrix row values");
    m.def("scatterd_add_gpu", &scatterd_add_gpu, "Implementation of scatter_add_ for doubles");
    m.def("scatteri_add_gpu", &scatteri_add_gpu, "Implementation of scatter_add_ for ints");
    m.def("rowselect_coo_gpu", &rowselect_coo_gpu, "COO row selection for sparsity-aware spgemm");
    m.def("coogeam_gpu", &coogeam_gpu, "csrgeam2 wrapper for cusparse");
    m.def("rowselect_csr_gpu", &rowselect_csr_gpu, "CSR row selection for sparsity-aware spgemm");
}

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/native/SparseTensorUtils.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include "cusparse.h"

#include <pybind11/pybind11.h>
typedef float real;
#include "nsparse_asm.h"

// #include <THC/THCGeneral.hpp>

#include <torch/extension.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

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

// NSPARSE defs
#define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)
#define WARP_BIT 5
#define WARP 32
#define MAX_LOCAL_THREAD_NUM 1024
#define MAX_THREAD_BLOCK (MAX_LOCAL_THREAD_NUM / WARP)
#define BIN_NUM 7
#define PWARP 4
#define IMB_PWMIN 32
#define B_PWMIN 16
#define IMB_MIN 512
#define B_MIN 256
#define IMB_PW_SH_SIZE 4096
#define B_PW_SH_SIZE 2048
#define IMB_SH_SIZE 1024
#define B_SH_SIZE 512
// #define HASH_SCAL 107 // Set disjoint number to COMP_SH_SIZE
#define HASH_SCAL 4 // Set disjoint number to COMP_SH_SIZE

/* Structure for SpGEMM */
typedef struct {
    cudaStream_t *stream;
    int *bin_size;
    int *bin_offset;
    int *d_bin_size;
    int *d_bin_offset;
    int *d_row_nz;
    int *d_row_perm;
    int max_intprod;
    int max_nz;
    int *d_max;
} sfBIN;

void init_bin(sfBIN *bin, int M)
{
    int i;
    bin->stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BIN_NUM);
    for (i = 0; i < BIN_NUM; i++) {
        cudaStreamCreate(&(bin->stream[i]));
    }
  
    bin->bin_size = (int *)malloc(sizeof(int) * BIN_NUM);
    bin->bin_offset = (int *)malloc(sizeof(int) * BIN_NUM);
    cudaMalloc((void **)&(bin->d_row_perm), sizeof(int) * M);
    cudaMalloc((void **)&(bin->d_row_nz), sizeof(int) * (M + 1));
    cudaMalloc((void **)&(bin->d_max), sizeof(int));
    cudaMalloc((void **)&(bin->d_bin_size), sizeof(int) * BIN_NUM);
    cudaMalloc((void **)&(bin->d_bin_offset), sizeof(int) * BIN_NUM);

    // auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    // torch::Tensor d_row_perm_tens = torch::empty({M}, torch::TensorOptions()
    //                                                                 .dtype(torch::kInt32)
    //                                                                 .device(torch::kCUDA));
    // torch::Tensor d_row_nz_tens = torch::zeros({M + 1}, options);
    // torch::Tensor d_max_tens = torch::zeros({1}, options);
    // torch::Tensor d_bin_size_tens = torch::zeros({BIN_NUM}, options);
    // torch::Tensor d_bin_offset_tens = torch::zeros({BIN_NUM}, options);

    // bin->d_row_perm = d_row_perm_tens.data<int>();
    // bin->d_row_nz = d_row_nz_tens.data<int>();
    // bin->d_max = d_max_tens.data<int>();
    // bin->d_bin_size = d_bin_size_tens.data<int>();
    // bin->d_bin_offset = d_bin_offset_tens.data<int>();

    i = 0;
    bin->max_intprod = 0;
    bin->max_nz = 0;

}

void release_bin(sfBIN bin)
{
    int i;
    cudaFree(bin.d_row_perm);
    cudaFree(bin.d_row_nz);
    cudaFree(bin.d_max);
    cudaFree(bin.d_bin_size);
    cudaFree(bin.d_bin_offset);

    free(bin.bin_size);
    free(bin.bin_offset);
    for (i = 0; i < BIN_NUM; i++) {
        cudaStreamDestroy(bin.stream[i]);
    }
    free(bin.stream);
}

__global__ void set_intprod_num(int *d_arpt, int *d_acol,
                                const int* __restrict__ d_brpt,
                                int *d_row_intprod, int *d_max_intprod,
                                int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = 0;
    int j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        nz_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_row_intprod[i] = nz_per_row;
    atomicMax(d_max_intprod, nz_per_row);
}

__global__ void set_bin(int *d_row_nz, int *d_bin_size, int *d_max,
                        int M, int min, int mmin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = d_row_nz[i];

    atomicMax(d_max, nz_per_row);

    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= (mmin)) {
                atomicAdd(d_bin_size + j, 1);
            }
            else {
                atomicAdd(d_bin_size + j + 1, 1);
            }
            return;
        }
    }
    atomicAdd(d_bin_size + BIN_NUM - 1, 1);
}

__global__ void init_row_perm(int *d_permutation, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }
  
    d_permutation[i] = i;
}

__global__ void set_row_perm(int *d_bin_size, int *d_bin_offset,
                             int *d_max_row_nz, int *d_row_perm,
                             int M, int min, int mmin)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) {
        return;
    }

    int nz_per_row = d_max_row_nz[i];
    int dest;
  
    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= mmin) {
                dest = atomicAdd(d_bin_size + j, 1);
                d_row_perm[d_bin_offset[j] + dest] = i;
            }
            else {
                dest = atomicAdd(d_bin_size + j + 1, 1);
                d_row_perm[d_bin_offset[j + 1] + dest] = i;
            }
            return;
        }
    }
    dest = atomicAdd(d_bin_size + BIN_NUM - 1, 1);
    d_row_perm[d_bin_offset[BIN_NUM - 1] + dest] = i;

}

void set_max_bin(int *d_arpt, int *d_acol, int *d_brpt, sfBIN *bin, int M)
{
    int i;
    int GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
  
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_intprod), sizeof(int), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_intprod_num<<<GS, BS>>>(d_arpt, d_acol, d_brpt, bin->d_row_nz, bin->d_max, M);
    cudaMemcpy(&(bin->max_intprod), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);

    if (bin->max_intprod > IMB_PWMIN) {
        set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size, bin->d_max, M, IMB_MIN, IMB_PWMIN);
  
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, IMB_MIN, IMB_PWMIN);
    }
    else {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_offset[i] = M;
        }
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
    }
}


void set_min_bin(sfBIN *bin, int M)
{
    int i;
    int GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
  
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_nz), sizeof(int), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size,
                        bin->d_max,
                        M, B_MIN, B_PWMIN);
  
    cudaMemcpy(&(bin->max_nz), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
    if (bin->max_nz > B_PWMIN) {
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
  
        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, B_MIN, B_PWMIN);
    }

    else {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
    }
}

__global__ void init_value(real *d_val, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) {
        return;
    }
    d_val[i] = 0;
}

__global__ void init_check(int *d_check, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) {
        return;
    }
    d_check[i] = -1;
}

__global__ void set_row_nz_bin_pwarp(const int *d_arpt, const int *d_acol,
                                     const int* __restrict__ d_brpt,
                                     const int* __restrict__ d_bcol,
                                     const int *d_row_perm,
                                     int *d_row_nz,
                                     int bin_offset, int M) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
  
    int j, k;
    int soffset;
    int acol, bcol, key, hash, adr, nz, old;
    __shared__ int check[IMB_PW_SH_SIZE];
  
    soffset = local_rid * IMB_PWMIN;
  
    for (j = tid; j < IMB_PWMIN; j += PWARP) {
        check[soffset + j] = -1;
    }
    if (rid >= M) {
        return;
    }

    rid = d_row_perm[rid + bin_offset];
    nz = 0;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (IMB_PWMIN - 1);
            adr = soffset + hash;
            while (1) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (IMB_PWMIN - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = PWARP / 2; j >= 1; j /= 2) {
        // nz += __shfl_xor(nz, j);
        nz += __shfl_xor_sync(0xFFFFFFFF, nz, j);
    }

    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}


template <int SH_ROW>
__global__ void set_row_nz_bin_each(const int *d_arpt, const int *d_acol,
                                    const int* __restrict__ d_brpt,
                                    const int* __restrict__ d_bcol,
                                    const int *d_row_perm,
                                    int *d_row_nz, int bin_offset, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / WARP;
    int tid = i % WARP;
    int wid = rid % (blockDim.x / WARP);
    int j, k, l;
    int bcol, key, hash, old;
    int nz, adr;
    int acol, ccol;
    int soffset;

    soffset = wid * SH_ROW;

    __shared__ int check[IMB_SH_SIZE];
    for (j = tid; j < SH_ROW; j += WARP) {
        check[soffset + j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    acol = 0;
    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid]; j < d_arpt[rid + 1]; j += WARP) {
        if (j + tid < d_arpt[rid + 1]) acol = ld_gbl_int32(d_acol + j + tid);
        for (l = 0; l < WARP && j + l < d_arpt[rid + 1]; l++) {
            ccol = __shfl(acol, l);
            for (k = d_brpt[ccol] + tid; k < d_brpt[ccol + 1]; k += WARP) {
                bcol = d_bcol[k];
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = soffset + hash;
                while (1) {
                    if (check[adr] == key) {
                        break;
                    }
                    else if (check[adr] == -1) {
                        old = atomicCAS(check + adr, -1, key);
                        if (old == -1) {
                            nz++;
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = soffset + hash;
                    }
                }
            }
        }
    }

    for (j = WARP / 2; j >= 1; j /= 2) {
        // nz += __shfl_xor(nz, j);
        nz += __shfl_xor_sync(0xFFFFFFFF, nz, j);
    }

    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol,
                                       int *d_row_perm, int *d_row_nz,
                                       int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;

    __shared__ int check[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        check[j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();

    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (1) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                }
            }
        }
    }

    for (j = WARP / 2; j >= 1; j /= 2) {
        // nz += __shfl_xor(nz, j);
        nz += __shfl_xor_sync(0xFFFFFFFF, nz, j);
    }
  
    __syncthreads();
    if (threadIdx.x == 0) {
        check[0] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(check, nz);
    }
    __syncthreads();
  
    if (threadIdx.x == 0) {
        d_row_nz[rid] = check[0];
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb_large(const int *d_arpt, const int *d_acol,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol,
                                             int *d_row_perm, int *d_row_nz,
                                             int *d_fail_count, int *d_fail_perm,
                                             int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int adr;
    int acol;

    __shared__ int check[SH_ROW];
    __shared__ int snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        check[j] = -1;
    }
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();
  
    rid = d_row_perm[rid + bin_offset];
    int count = 0;
    int border = SH_ROW >> 1;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (count < border && snz[0] < border) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(snz, 1);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                    count++;
                }
            }
            if (count >= border || snz[0] >= border) {
                break;
            }
        }
        if (count >= border || snz[0] >= border) {
            break;
        }
    }
  
    __syncthreads();
    if (count >= border || snz[0] >= border) {
        if (threadIdx.x == 0) {
            int d = atomicAdd(d_fail_count, 1);
            d_fail_perm[d] = rid;
        }
    }
    else {
        if (threadIdx.x == 0) {
            d_row_nz[rid] = snz[0];
        }
    }
}

__global__ void set_row_nz_bin_each_gl(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol,
                                       const int *d_row_perm,
                                       int *d_row_nz, int *d_check,
                                       int max_row_nz, int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;
    int offset = rid * max_row_nz;

    __shared__ int snz[1];
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
    __syncthreads();
  
    if (rid >= M) {
        return;
    }
  
    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = offset + hash;
            while (1) {
                if (d_check[adr] == key) {
                    break;
                }
                else if (d_check[adr] == -1) {
                    old = atomicCAS(d_check + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = offset + hash;
                }
            }
        }
    }
  
    for (j = WARP / 2; j >= 1; j /= 2) {
        // nz += __shfl_xor(nz, j);
        nz += __shfl_xor_sync(0xFFFFFFFF, nz, j);
    }
  
    if (tid == 0) {
        atomicAdd(snz, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        d_row_nz[rid] = snz[0];
    }
}

void set_row_nnz(int *d_arpt, int *d_acol,
                 int *d_brpt, int *d_bcol,
                 int *d_crpt,
                 sfBIN *bin,
                 int M, int *nnz);
  

__global__ void calculate_value_col_bin_pwarp(const int *d_arpt,
                                              const int *d_acol,
                                              const real *d_aval,
                                              const int* __restrict__ d_brpt,
                                              const int* __restrict__ d_bcol,
                                              const real* __restrict__ d_bval,
                                              int *d_crpt,
                                              int *d_ccol,
                                              real *d_cval,
                                              const int *d_row_perm,
                                              int *d_nz,
                                              int bin_offset,
                                              int bin_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
    int j;
    __shared__ int shared_check[B_PW_SH_SIZE];
    __shared__ real shared_value[B_PW_SH_SIZE];
  
    int soffset = local_rid * (B_PWMIN);
  
    for (j = tid; j < (B_PWMIN); j += PWARP) {
        shared_check[soffset + j] = -1;
        shared_value[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_row_perm[rid + bin_offset];
  
    if (tid == 0) {
        d_nz[rid] = 0;
    }
    int k;
    int acol, bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcol[k];
            bval = d_bval[k];
	
            key = bcol;
            hash = (bcol * HASH_SCAL) & ((B_PWMIN) - 1);
            adr = soffset + hash;
            while (1) {
                if (shared_check[adr] == key) {
                    atomic_fadd(shared_value + adr, aval * bval);
                    break;
                }
                else if (shared_check[adr] == -1) {
                    old = atomicCAS(shared_check + adr, -1, key);
                    if (old == -1) {
                        atomic_fadd(shared_value + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & ((B_PWMIN) - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = tid; j < (B_PWMIN); j += PWARP) {
        if (shared_check[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            shared_check[soffset + index] = shared_check[soffset + j];
            shared_value[soffset + index] = shared_value[soffset + j];
        }
    }
    int nz = d_nz[rid];
    // Sorting for shared data
    int count, target;
    for (j = tid; j < nz; j += PWARP) {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[soffset + j];
        d_cval[offset + count] = shared_value[soffset + j];
    }
}


template <int SH_ROW>
__global__ void calculate_value_col_bin_each(const int *d_arpt,
                                             const int *d_acol,
                                             const real *d_aval,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol,
                                             const real* __restrict__ d_bval,
                                             int *d_crpt,
                                             int *d_ccol,
                                             real *d_cval,
                                             const int *d_row_perm,
                                             int *d_nz,
                                             int bin_offset,
                                             int bin_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / WARP;
    int tid = i % WARP;
    int wid = rid % (blockDim.x / WARP);
    int j;
    __shared__ int shared_check[B_SH_SIZE];
    __shared__ real shared_value[B_SH_SIZE];
  
    int soffset = wid * SH_ROW;

    for (j = tid; j < SH_ROW; j += WARP) {
        shared_check[soffset + j] = -1;
        shared_value[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_row_perm[rid + bin_offset];

    if (tid == 0) {
        d_nz[rid] = 0;
    }
    int lacol, acol;
    int k, l;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real laval, aval, bval;

    lacol = 0;
    for (j = d_arpt[rid]; j < d_arpt[rid + 1]; j += WARP) {
        if (j + tid < d_arpt[rid + 1]) {
            lacol = ld_gbl_int32(d_acol + j + tid);
            laval = ld_gbl_real(d_aval + j + tid);
        }
        for (l = 0; l < WARP && j + l < d_arpt[rid + 1]; l++) {
            acol = __shfl(lacol, l);
            aval = __shfl(laval, l);
            for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
                bcol = d_bcol[k];
                bval = d_bval[k];
	
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = soffset + hash;
                while (1) {
                    if (shared_check[adr] == key) {
                        atomic_fadd(shared_value + adr, aval * bval);
                        break;
                    }
                    else if (shared_check[adr] == -1) {
                        old = atomicCAS(shared_check + adr, -1, key);
                        if (old == -1) {
                            atomic_fadd(shared_value + adr, aval * bval);
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = soffset + hash;
                    }
                }
            }
        }
    }
  
    for (j = tid; j < SH_ROW; j += WARP) {
        if (shared_check[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            shared_check[soffset + index] = shared_check[soffset + j];
            shared_value[soffset + index] = shared_value[soffset + j];
        }
    }
    int nz = d_nz[rid];
    /* Sorting for shared data */
    int count, target;
    for (j = tid; j < nz; j += WARP) {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[soffset + j];
        d_cval[offset + count] = shared_value[soffset + j];
    }
}

template <int SH_ROW>
__global__ void calculate_value_col_bin_each_tb(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int bin_offset,
                                                int bin_size)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j;
    __shared__ int shared_check[SH_ROW];
    __shared__ real shared_value[SH_ROW];
  
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        shared_check[j] = -1;
        shared_value[j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }

    rid = d_row_perm[rid + bin_offset];

    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            bval = d_bval[k];
	
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            while (1) {
                if (shared_check[hash] == key) {
                    atomic_fadd(shared_value + hash, aval * bval);
                    break;
                }
                else if (shared_check[hash] == -1) {
                    old = atomicCAS(shared_check + hash, -1, key);
                    if (old == -1) {
                        atomic_fadd(shared_value + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < WARP) {
        for (j = tid; j < SH_ROW; j += WARP) {
            if (shared_check[j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                shared_check[index] = shared_check[j];
                shared_value[index] = shared_value[j];
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = shared_check[j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[j];
        d_cval[offset + count] = shared_value[j];
    }

}

__global__ void calculate_value_col_bin_each_gl(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int *d_check,
                                                real *d_value,
                                                int max_row_nz,
                                                int bin_offset,
                                                int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j;
  
    if (rid >= M) {
        return;
    }

    int doffset = rid * max_row_nz;

    rid = d_row_perm[rid + bin_offset];
  
    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WARP) {
            bcol = d_bcol[k];
            bval = d_bval[k];
      
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = doffset + hash;
            while (1) {
                if (d_check[adr] == key) {
                    atomic_fadd(d_value + adr, aval * bval);
                    break;
                }
                else if (d_check[adr] == -1) {
                    old = atomicCAS(d_check + adr, -1, key);
                    if (old == -1) {
                        atomic_fadd(d_value + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = doffset + hash;
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < WARP) {
        for (j = tid; j < max_row_nz; j += WARP) {
            if (d_check[doffset + j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                d_check[doffset + index] = d_check[doffset + j];
                d_value[doffset + index] = d_value[doffset + j];
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];
  
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = d_check[doffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(d_check[doffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = d_check[doffset + j];
        d_cval[offset + count] = d_value[doffset + j];
    }

}

void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,
                             int *d_brpt, int *d_bcol, real *d_bval,
                             int *d_crpt, int *d_ccol, real *d_cval,
                             sfBIN *bin,
                             int M);
  
void set_row_nnz(int *d_arpt, int *d_acol,
                 int *d_brpt, int *d_bcol,
                 int *d_crpt,
                                      sfBIN *bin,
                 int M,
                 int *nnz)
{
                                                int i;
    int GS, BS;
    for (i = BIN_NUM - 1; i >= 0; i--) {
        if (bin->bin_size[i] > 0) {
            switch (i) {
            case 0:
                BS = 512;
                GS = div_round_up(bin->bin_size[i] * PWARP, BS);
                set_row_nz_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>
                    (d_arpt, d_acol,
                    d_brpt, d_bcol,
                                         bin->d_row_perm,
                    bin->d_row_nz,
                    bin->bin_offset[i],
                    bin->bin_size[i]);
                break;
            case 1 :
                             	BS = 64;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<512><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 2 :
                             	BS = 128;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<1024><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 3 :
                             	BS = 256;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<2048><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 4 :
                             	BS = 512;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<4096><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 5 :
                             	BS = 1024;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<8192><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 6 :
                                     	{
            	    int fail_count;
            	    fail_count = 0;
            	    int *d_fail_count, *d_fail_perm;
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(int)));
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_perm, sizeof(int) * bin->bin_size[i]));
            	    cudaMemcpy(d_fail_count, &fail_count, sizeof(int), cudaMemcpyHostToDevice);
            	    BS = 1024;
            	    GS = bin->bin_size[i];
            	    set_row_nz_bin_each_tb_large<8192><<<GS, BS, 0, bin->stream[i]>>>
            	      (d_arpt, d_acol, d_brpt, d_bcol,
            	       bin->d_row_perm, bin->d_row_nz,
            	       d_fail_count, d_fail_perm,
            	       bin->bin_offset[i], bin->bin_size[i]);
            	    cudaMemcpy(&fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);
            	    if (fail_count > 0) {
              	        int max_row_nz = bin->max_intprod;
            	        size_t table_size = (size_t)max_row_nz * fail_count;
            	        int *d_check;
            	        checkCudaErrors(cudaMalloc((void **)&(d_check), sizeof(int) * table_size));

            	        BS = 1024;
            	        GS = div_round_up(table_size, BS);
            	        init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);
            	        GS = bin->bin_size[i];
	                    set_row_nz_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>
                     		  (d_arpt, d_acol, d_brpt, d_bcol,
		                   d_fail_perm, bin->d_row_nz, d_check,
             		                   max_row_nz, 0, fail_count);
                                  	                    cudaFree(d_check);
  	                }
	            cudaFree(d_fail_count);
	            cudaFree(d_fail_perm);
	        }
	        break;
	      default :
	          exit(0);
	      }
        }
      }
      cudaThreadSynchronize();

    /* Set row pointer of matrix C */
    thrust::exclusive_scan(thrust::device, bin->d_row_nz, bin->d_row_nz + (M + 1), d_crpt, 0);
    cudaMemcpy(nnz, d_crpt + M, sizeof(int), cudaMemcpyDeviceToHost);
}

void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,
			     int *d_brpt, int *d_bcol, real *d_bval,
			     int *d_crpt, int *d_ccol, real *d_cval,
			     sfBIN *bin,
			     int M)
{
                       int i;
  int GS, BS;
  for (i = BIN_NUM - 1; i >= 0; i--) {
    if (bin->bin_size[i] > 0) {
      switch (i) {
      case 0:
      BS = 512;
      GS = div_round_up(bin->bin_size[i] * PWARP, BS);
      calculate_value_col_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>
           (d_arpt, d_acol, d_aval,
	   d_brpt, d_bcol, d_bval,
	   d_crpt, d_ccol, d_cval,
	   bin->d_row_perm, bin->d_row_nz,
	   bin->bin_offset[i], bin->bin_size[i]);
      break;
      case 1:
	  BS = 64;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<256><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 2:
	  BS = 128;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<512><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 3:
	  BS = 256;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<1024><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 4:
	  BS = 512;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<2048><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 5:
	  BS = 1024;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<4096><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
	case 6 :
	  {
	    int max_row_nz = bin->max_nz * 2;
	    int table_size = max_row_nz * bin->bin_size[i];
	    int *d_check;
	    real *d_value;
	    checkCudaErrors(cudaMalloc((void **)&(d_check), sizeof(int) * table_size));
	    checkCudaErrors(cudaMalloc((void **)&(d_value), sizeof(real) * table_size));

	    BS = 1024;
	    GS = div_round_up(table_size, BS);
	    init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);
	    init_value<<<GS, BS, 0, bin->stream[i]>>>(d_value, table_size);
	    GS = bin->bin_size[i];
	    calculate_value_col_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>
	      (d_arpt, d_acol, d_aval,
	       d_brpt, d_bcol, d_bval,
	       d_crpt, d_ccol, d_cval,
	       bin->d_row_perm, bin->d_row_nz,
	       d_check, d_value, max_row_nz,
	       bin->bin_offset[i], bin->bin_size[i]);
	    cudaFree(d_check);
	    cudaFree(d_value);
	  }
	  break;
	default :
	  exit(0);
	}
      }
    }
  cudaThreadSynchronize();
}

__device__ int binary_searchd(double *arr, double val, int imin, int imax) {
    
    int ans = 0;
    while (imax >= imin) {
        int imid = (imin + imax) / 2;
        // printf("imin: %d imax: %d imid: %d\n", imin, imax, imid);

        if (arr[imid] < val) {
            imin = imid + 1;
        } else {
            ans = imid;
            imax = imid - 1;
        }
    }
    
    return ans;
}

__device__ int binary_searchf(float *arr, float val, int imin, int imax) {
    
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

__device__ long binary_searchi(int *arr, int val, int imin, int imax) {
    
    int ans = -1;
    while (imax >= imin) {
        int imid = (imin + imax) / 2;

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

at::Tensor sparse_csr_tensor_gpu(const at::Tensor& crows, 
                                    const at::Tensor& cols, 
                                    const at::Tensor& values, 
                                    at::ArrayRef<int64_t> size) {

    // return at::sparse_compressed_tensor(crows, cols, values, size, c10::Layout::SparseCsr); // torch 1.13
    return at::sparse_csr_tensor(crows, cols, values, size, values.options().layout(at::kSparseCsr)); // 1.11
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
                                                // CUSPARSE_CSRMM_ALG1,                // alg
                                                CUSPARSE_SPMM_CSR_ALG1,                // alg
                                                &bufferSize));                      // bufferSize

    void* d_buffer = NULL;
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
                                    // CUSPARSE_CSRMM_ALG1,                // alg
                                    CUSPARSE_SPMM_CSR_ALG1,                // alg
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

std::vector<at::Tensor> nsparse_spgemm(
        const at::Tensor& A_crow_indices, 
        const at::Tensor& A_col_indices,
        const at::Tensor& A_values, 
        const at::Tensor& B_crow_indices, 
        const at::Tensor& B_col_indices,
        const at::Tensor& B_values,
        int32_t n,
        int32_t k,
        int32_t m) {

    sfBIN bin;

    /* Set max bin */
    int *d_a_crow_indices = A_crow_indices.data<int>();
    int *d_a_col_indices = A_col_indices.data<int>();
    float *d_a_values = A_values.data<float>();

    int *d_b_crow_indices = B_crow_indices.data<int>();
    int *d_b_col_indices = B_col_indices.data<int>();
    float *d_b_values = B_values.data<float>();

    init_bin(&bin, n);
    std::cout << "after init_bin" << std::endl;

    // int *d_c_crow_indices = C_crow_indices.data<int>();
    // int *d_c_crow_indices;
    // cudaMalloc(&d_c_crow_indices, sizeof(int) * (n + 1));
    torch::Tensor C_crow_indices = torch::zeros({n + 1}, torch::TensorOptions()
                                                                    .dtype(torch::kInt32)
                                                                    .device(torch::kCUDA));
    int *d_c_crow_indices = C_crow_indices.data<int>();

    set_max_bin(d_a_crow_indices, d_a_col_indices, d_b_crow_indices, &bin, n);
    std::cout << "after set_max_bin" << std::endl;
  
    int C_nnz;
    /* Count nz of C */
    set_row_nnz(d_a_crow_indices, d_a_col_indices,
                d_b_crow_indices, d_b_col_indices,
                d_c_crow_indices,
                &bin,
                n,
                &C_nnz);
    std::cout << "after set_row_nnz" << std::endl;

    /* Set bin */
    set_min_bin(&bin, n);
    std::cout << "after set_min_bin" << std::endl;
  
    // int *d_c_col_indices;
    // float *d_c_values;
    // cudaMalloc(&d_c_col_indices, sizeof(int) * C_nnz);
    // cudaMalloc(&d_c_values, sizeof(float) * C_nnz);
    torch::Tensor C_col_indices = torch::zeros({C_nnz}, torch::TensorOptions()
                                                                    .dtype(torch::kInt32)
                                                                    .device(torch::kCUDA));

    torch::Tensor C_values = torch::zeros({C_nnz}, torch::TensorOptions()
                                                                    .dtype(torch::kFloat32)
                                                                    .device(torch::kCUDA));
    int *d_c_col_indices = C_col_indices.data<int>();
    float *d_c_values = C_values.data<float>();
  
    /* Calculating value of C */
    calculate_value_col_bin(d_a_crow_indices, d_a_col_indices, d_a_values,
                            d_b_crow_indices, d_b_col_indices, d_b_values,
                            d_c_crow_indices, d_c_col_indices, d_c_values,
                            &bin,
                            n);
    std::cout << "after calculate_value_col_bin" << std::endl;

    release_bin(bin);
    std::cout << "after release_bin" << std::endl;

    CHECK_ERROR("nsparse_spgemm")

    return {C_crow_indices, C_col_indices, C_values};
    // return {c_csr_offsets, c_columns, c_values};
}

std::vector<at::Tensor> spgemm_gpu(
        const at::Tensor& A_crow_indices, 
        const at::Tensor& A_col_indices,
        const at::Tensor& A_values, 
        const at::Tensor& B_crow_indices, 
        const at::Tensor& B_col_indices,
        const at::Tensor& B_values,
        const at::Tensor& C_crow_indices, 
        const at::Tensor& C_col_indices,
        const at::Tensor& C_values,
        int32_t n,
        int32_t k,
        int32_t m) {

    
    // SpGEMM between A (n x k)  and B (k x m)
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    int32_t A_nnz = A_values.size(0);
    int32_t B_nnz = B_values.size(0);
    int32_t C_nnz = C_values.size(0);

    float alpha = 1;
    float beta = 1;

    // Construct CSR matrix structs for A, B, and C (empty CSR)
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
					  n, 		// rows
					  k, 	        // cols
					  A_nnz, 	// nnz
					  A_crow_indices.data<int>(), // csrRowOffsets
					  A_col_indices.data<int>(), // csrColInd
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
					  B_crow_indices.data<int>(), 	// csrRowOffsets
					  B_col_indices.data<int>(), // csrColInd
					  B_values.data<float>(),  // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType

    cusparseSpMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matC,
					  n, 		// rows
					  m, 	        // cols
					  C_nnz, 	// nnz
					  C_crow_indices.data<int>(), 	// csrRowOffsets
					  C_col_indices.data<int>(),    // csrColInd
					  C_values.data<float>(),       // csrValues
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
// __global__ void ComputeDarts1D(float *dart_values, float *p_rowsum, float *ps_p_rowsum, 
                                    long *ps_dart_count_rows, int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int dart_count = n_darts * mb_count;
    for (int i = id; i < n_darts; i += stride) {
        // int row = i / n_darts;
        long row = binary_searchl(ps_dart_count_rows, i, 0, mb_count);
        dart_values[i] *= p_rowsum[row];
        dart_values[i] += ps_p_rowsum[row];
    }
}

void compute_darts1d_gpu(const at::Tensor& dart_values, const at::Tensor& p_rowsum, 
				const at::Tensor& ps_p_rowsum, const at::Tensor& ps_dart_count_rows, 
                                int n_darts, int mb_count) {

    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(n_darts / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ComputeDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<double>(), 
                                                p_rowsum.data<double>(),
                                                ps_p_rowsum.data<double>(),
    // ComputeDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<float>(), 
    //                                             p_rowsum.data<float>(),
    //                                             ps_p_rowsum.data<float>(),
                                                ps_dart_count_rows.data<long>(),
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
        // int vtx = binary_searchf(ps_p_values, dart_values[i], 0, nnz - 1);
        int vtx = binary_searchd(ps_p_values, dart_values[i], 0, nnz - 1);
        // printf("i: %d vtx: %d\n", i, vtx);
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

    // ThrowDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<double>(), 
    //                                             ps_p_values.data<double>(), 
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
        // int vtx = binary_searchf(ps_dart_hits_inv, dart_select[i], 0, nnz);
        int vtx = binary_searchd(ps_dart_hits_inv, dart_select[i], 0, nnz);
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

__global__ void Normalized(double *output, double *input, long *index, int len) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < len; i += stride) {
        output[i] /= input[index[i]];
    }
}

void normalized_gpu(const at::Tensor& output, const at::Tensor& input, const at::Tensor& index, int len) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    Normalized<<<BLOCK_COUNT, BLOCK_SIZE>>>(output.data<double>(), input.data<double>(), index.data<long>(), len);
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
}

__global__ void NormalizeCsr(double *values, long *crows, int len) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < len; i += stride) {
        // output[i] /= input[index[i]];
        int sum = 0;
        for (int j = crows[i]; j < crows[i + 1]; j++) {
            sum += values[j];
        }
        for (int j = crows[i]; j < crows[i + 1]; j++) {
            values[j] /= sum;
        }
    }
}

void normalize_csr_gpu(const at::Tensor& values, const at::Tensor& crows, int len) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    NormalizeCsr<<<BLOCK_COUNT, BLOCK_SIZE>>>(values.data<double>(), crows.data<long>(), len);
}

__global__ void SumCsrd(double *output, long *crows, double *values, int len) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < len; i += stride) {
        // output[i] /= input[index[i]];
        for (int j = crows[i]; j < crows[i + 1]; j++) {
            output[i] += values[j];
        }
    }
}

void sum_csrd_gpu(const at::Tensor& output, const at::Tensor& crows, const at::Tensor& values, int len) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    SumCsrd<<<BLOCK_COUNT, BLOCK_SIZE>>>(output.data<double>(), crows.data<long>(), values.data<double>(), len);
}

__global__ void SumCsri(int *output, long *crows, int *values, int len) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < len; i += stride) {
        // output[i] /= input[index[i]];
        for (int j = crows[i]; j < crows[i + 1]; j++) {
            output[i] += values[j];
        }
    }
}

void sum_csri_gpu(const at::Tensor& output, const at::Tensor& crows, const at::Tensor& values, int len) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    SumCsri<<<BLOCK_COUNT, BLOCK_SIZE>>>(output.data<int>(), crows.data<long>(), values.data<int>(), len);
}

__global__ void NormalizeBatch(float *output, int *input, int output_rows, int output_cols) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < output_rows * output_cols; i += stride) {
        int row = i / output_cols;
        output[i] /= input[row];
    }
}

void normalize_batch_gpu(const at::Tensor& output, const at::Tensor& input, int rows, int cols) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((rows * cols) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    NormalizeBatch<<<BLOCK_COUNT, BLOCK_SIZE>>>(output.data<float>(), input.data<int>(), rows, cols);
    CHECK_ERROR("normalize batch error")
}

__global__ void ShiftRowSelect(long *row_shift, long *row_select_rows, int rank, int size,
                                    int replication, int nnz, int batch_size, int node_count, int mb_count,
                                    int shift_size, int semibulk_size) { 

    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    int rank_c = rank / replication;
    long proc_row_chunk = rank_c * ((batch_size * mb_count) / (size / replication));
    
    for (int i = id; i < nnz; i += stride) {
        long mb_row = (row_select_rows[i] + proc_row_chunk) / shift_size;
        // row_shift[i] += mb_row * node_count;
        row_shift[i] += (mb_row % semibulk_size) * node_count;
    }
}

void shift_rowselect_gpu(const at::Tensor& row_shift, const at::Tensor& row_select_rows,
                            int nnz, int rank, int size, int replication, int batch_size, int node_count, 
                            int mb_count, int shift_size, int semibulk_size) {


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
                                                    shift_size,
                                                    semibulk_size);
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

// __global__ void ScatterAddD(double *src, long *indices, double *values, int num_vals) {
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
// 
//     for (int i = id; i < num_vals; i += stride) {
//         atomicAdd(&src[indices[i]], values[i]);
//     } 
// }
// 
// void scatterd_add_gpu(const at::Tensor& src, const at::Tensor& indices, const at::Tensor& values, int num_vals) {
// 
// 
//     int BLOCK_SIZE = 256;
//     int BLOCK_COUNT = std::ceil(num_vals / ((float) BLOCK_SIZE));
//     BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);
// 
//     ScatterAddD<<<BLOCK_COUNT, BLOCK_SIZE>>>(src.data<double>(), 
//                                                 indices.data<long>(), 
//                                                 values.data<double>(),
//                                                 num_vals);
//     CHECK_ERROR("scatter add doubles error")
// }

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

__global__ void VtxTally(int *proc_tally, long *vtxs, int vtxs_count, int nodes_per_proc, int proc_count) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < vtxs_count; i += stride) {
        int vtx = (int) vtxs[i];
        int dst_proc = vtx / nodes_per_proc;
        if (dst_proc >= proc_count) {
            dst_proc = proc_count - 1;
        }
        atomicAdd(&proc_tally[dst_proc], 1);
    } 
}

__global__ void SortVtxs(long *vtxs, long *src_vtx_send, int *ps_proc_tally, long *og_idx, int vtxs_count, 
                            int nodes_per_proc, int proc_count) { 

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < vtxs_count; i += stride) {
        long vtx = vtxs[i];
        int dst_proc = vtx / nodes_per_proc; 
        if (dst_proc >= proc_count) {
            dst_proc = proc_count - 1;
        }
        int idx = atomicAdd(&ps_proc_tally[dst_proc], 1);
        src_vtx_send[idx] = vtx;
        og_idx[idx] = i;
    } 
}

void sort_dst_proc_gpu(const at::Tensor& vtxs, const at::Tensor& src_vtx_sort, const at::Tensor& og_idxs, 
                            const at::Tensor& tally, int nodes_per_proc, int proc_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(vtxs.sizes()[0] / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (nodes_per_proc == 0 || vtxs.sizes()[0] == 0) {
        return;
    }

    VtxTally<<<BLOCK_COUNT, BLOCK_SIZE>>>(tally.data<int>(), 
                                            vtxs.data<long>(), 
                                            vtxs.sizes()[0], 
                                            nodes_per_proc,
                                            proc_count);

    auto ps_proc_tally = tally.cumsum(0, torch::kInt32).roll(1);
    ps_proc_tally[0] = 0;

    SortVtxs<<<BLOCK_COUNT, BLOCK_SIZE>>>(vtxs.data<long>(), 
                                            src_vtx_sort.data<long>(), 
                                            ps_proc_tally.data<int>(), 
                                            og_idxs.data<long>(), 
                                            vtxs.sizes()[0], 
                                            nodes_per_proc,
                                            proc_count);
    CHECK_ERROR("sort_dst_proc error")
}

__global__ void RearrangelRows(long *mata_rows, long *mata_cols, long *matc_crows, long *matb_crows, 
                                long *matb_cols, long *matc_cols, int nnz_count) { 

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < nnz_count; i += stride) {
        long row = mata_rows[i];
        long col = mata_cols[i];
        long dst_idx = matc_crows[row];
        long src_idx = matb_crows[col];
        long row_len = matb_crows[col + 1] - matb_crows[col];
        // memcpy(&matc_cols[dst_idx], &matb_cols[src_idx], row_len * sizeof(long));
        for (int j = 0; j < row_len; j++) {
            matc_cols[dst_idx + j] = matb_cols[src_idx + j];
        }
    } 
}

void rearrangel_rows_gpu(const at::Tensor& mata_rows, const at::Tensor& mata_cols, 
                            const at::Tensor& matc_crows, const at::Tensor& matb_crows, 
                            const at::Tensor& matb_cols, const at::Tensor& matc_cols) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(mata_rows.sizes()[0] / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (mata_rows.sizes()[0] == 0) {
        return;
    }

    RearrangelRows<<<BLOCK_COUNT, BLOCK_SIZE>>>(mata_rows.data<long>(), 
                                                    mata_cols.data<long>(),
                                                    matc_crows.data<long>(), 
                                                    matb_crows.data<long>(), 
                                                    matb_cols.data<long>(),
                                                    matc_cols.data<long>(),
                                                    mata_rows.sizes()[0]);

    CHECK_ERROR("rearrange_rows error")
}

__global__ void RearrangeRows(long *mata_rows, long *mata_cols, long *matc_crows, int *matb_crows, 
                                int *matb_cols, int *matc_cols, long nnz_count) { 

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < nnz_count; i += stride) {
        long row = mata_rows[i];
        long col = mata_cols[i];
        long dst_idx = matc_crows[row];
        long src_idx = matb_crows[col];
        long row_len = matb_crows[col + 1] - matb_crows[col];
        // memcpy(&matc_cols[dst_idx], &matb_cols[src_idx], row_len * sizeof(long));
        for (int j = 0; j < row_len; j++) {
            matc_cols[dst_idx + j] = (int)(matb_cols[src_idx + j]);
        }
    } 
}

void rearrange_rows_gpu(const at::Tensor& mata_rows, const at::Tensor& mata_cols, 
                            const at::Tensor& matc_crows, const at::Tensor& matb_crows, 
                            const at::Tensor& matb_cols, const at::Tensor& matc_cols) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(mata_rows.sizes()[0] / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (mata_rows.sizes()[0] == 0) {
        return;
    }

    RearrangeRows<<<BLOCK_COUNT, BLOCK_SIZE>>>(mata_rows.data<long>(), 
                                                    mata_cols.data<long>(),
                                                    matc_crows.data<long>(), 
                                                    matb_crows.data<int>(), 
                                                    matb_cols.data<int>(),
                                                    matc_cols.data<int>(),
                                                    mata_rows.sizes()[0]);

    CHECK_ERROR("rearrange_rows error")
}

// __global__ void ReduceSum(long *matc_crows, long *mata_crows, long *matb_crows, long *matc_cols, 
//                             long *mata_cols, long *matb_cols, int row_count) { 
// 
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
// 
//     for (int i = id; i < row_count; i += stride) {
//         int row_start = matc_crows[i];
//         int mata_row_len = mata_crows[i + 1] - mata_crows[i];
//         int matb_row_len = matb_crows[i + 1] - matb_crows[i];
//         if (mata_row_len == 0) {
//             // cudaMemcpyAsync(&matc_cols[row_start], &matc_cols[matb_crows[i]], matb_row_len * sizeof(long),
//             //                     cudaMemcpyDeviceToDevice);
//             for (int j = 0; j < matb_row_len; j++) {
//                 matc_cols[row_start + j] = matb_cols[matb_crows[i] + j];
//             }
//         } else if (matb_row_len == 0) {
//             // cudaMemcpyAsync(&matc_cols[row_start], &mata_cols[mata_crows[i]], mata_row_len * sizeof(long),
//             //                     cudaMemcpyDeviceToDevice);
//             for (int j = 0; j < mata_row_len; j++) {
//                 printf("");
//                 // printf("i: %d j: %d row_start: %d mata_col: %ld\n", i, j, row_start, mata_cols[mata_crows[i] + j]);
//                 matc_cols[row_start + j] = mata_cols[mata_crows[i] + j];
//             }
//         } 
//     } 
// }

__global__ void ReduceSum(long *matc_crows, long *mata_crows, 
                            long *matc_cols, long *mata_cols, 
                            int row_count) { 

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < row_count; i += stride) {
        int row_id = i;
        int row_len = mata_crows[row_id + 1] - mata_crows[row_id];
        for (int j = 0; j < row_len; j++) {
            matc_cols[matc_crows[row_id] + j] = mata_cols[mata_crows[row_id] + j];
        }
    } 

    // for (int i = id; i < row_count; i += stride) {
    //     int row_id = i;
    //     int row_len = matb_crows[row_id + 1] - matb_crows[row_id];
    //     for (int j = 0; j < row_len; j++) {
    //         matc_cols[matc_crows[row_id] + j] = matb_cols[matb_crows[row_id] + j];
    //     }
    // } 
}

void reduce_sum_gpu(const at::Tensor& matc_crows, const at::Tensor& mata_crows, 
                        const at::Tensor& matb_crows, const at::Tensor& matc_cols,
                        const at::Tensor& mata_cols, const at::Tensor& matb_cols,
                        const at::Tensor &mata_nnz_rows, 
                        const at::Tensor &matb_nnz_rows) {

    int BLOCK_SIZE = 256;
    // int mata_nnz_row_count = mata_nnz_rows.sizes()[0];
    // int matb_nnz_row_count = matb_nnz_rows.sizes()[0];
    // int ROW_COUNT = std::max(mata_nnz_row_count, matb_nnz_row_count);
    int ROW_COUNT = matc_crows.sizes()[0] - 1;
    int BLOCK_COUNT = std::ceil(ROW_COUNT / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (ROW_COUNT == 0) {
        return;
    }

    ReduceSum<<<BLOCK_COUNT, BLOCK_SIZE>>>(matc_crows.data<long>(), 
                                                mata_crows.data<long>(),
                                                matc_cols.data<long>(), 
                                                mata_cols.data<long>(),
                                                ROW_COUNT);

    ReduceSum<<<BLOCK_COUNT, BLOCK_SIZE>>>(matc_crows.data<long>(), 
                                                matb_crows.data<long>(),
                                                matc_cols.data<long>(), 
                                                matb_cols.data<long>(),
                                                ROW_COUNT);

    CHECK_ERROR("reduce sum error")
}

__global__ void ShadowColselect(long *sampled_frontiers, long *sampled_frontiers_rowids, 
                                    long *sampled_frontiers_crows, long *sampled_frontiers_cols,
                                    long *adj_crows, long *adj_cols, long *mask, long *ps_batch_sizes,
                                    int batch_size, int nnz_col_count) { 

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = id; i < nnz_col_count; i += stride) {
        long vtx = sampled_frontiers[i];
        long row_id = sampled_frontiers_rowids[i];

        long adj_start = adj_crows[i];
        long adj_stop = adj_crows[i + 1];
        long sampled_start = sampled_frontiers_crows[row_id];
        long sampled_stop = sampled_frontiers_crows[row_id + 1];
        long batch_id = row_id / batch_size;
        for (long j = adj_start; j < adj_stop; j++) {
            for (long k = sampled_start; k < sampled_stop; k++) {
                if (adj_cols[j] == sampled_frontiers_cols[k]) {
                    mask[j] = k - ps_batch_sizes[batch_id];
                    break;
                }
            }
        }
    } 
}


// shadow_colselect_gpu(sampled_frontiers, sampled_frontier_rowids, sampled_frontier_csr.crow_indices(),
//                         sampled_frontier_csr.col_indices(), row_select_adj.crow_indices(), colselect_mask, 
//                         sampled_frontiers.size(0), row_select_adj._nnz())
void shadow_colselect_gpu(const at::Tensor& sampled_frontiers, 
                            const at::Tensor& sampled_frontiers_rowids,
                            const at::Tensor& sampled_frontiers_crows, 
                            const at::Tensor& sampled_frontiers_cols,
                            const at::Tensor& adj_crows, 
                            const at::Tensor& adj_cols, 
                            const at::Tensor& colselect_mask, 
                            const at::Tensor& ps_batch_sizes,
                            int batch_size, int nnz_col_count, int nnz_count) {

    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz_col_count / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (nnz_col_count == 0) {
        return;
    }

    ShadowColselect<<<BLOCK_COUNT, BLOCK_SIZE>>>(sampled_frontiers.data<long>(), 
                                                    sampled_frontiers_rowids.data<long>(), 
                                                    sampled_frontiers_crows.data<long>(),
                                                    sampled_frontiers_cols.data<long>(),
                                                    adj_crows.data<long>(),
                                                    adj_cols.data<long>(),
                                                    // colselect_mask.data<bool>(), 
                                                    colselect_mask.data<long>(), 
                                                    ps_batch_sizes.data<long>(), 
                                                    batch_size,
                                                    nnz_col_count);

    fflush(stdout);
    CHECK_ERROR("shadow colselect error")
}


__global__ void RowSelectCsrDupes(long *nnz_cols, long *adj_crows, long *adj_cols, long *adj_vals, 
                                    long *rowselect_crows, long *rowselect_cols, long *rowselect_vals, 
                                    int nnz_col_count) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = id; i < nnz_col_count; i += stride) {
        long vtx = nnz_cols[i];
        long degree = adj_crows[vtx + 1] - adj_crows[vtx];
        // might be able to use memset here
        for (long j = 0; j < degree; j++) {
            // mask[row_offsets[vtx] + j] = true;
            rowselect_cols[rowselect_crows[i] + j] = adj_cols[adj_crows[vtx] + j];
            // rowselect_vals[rowselect_crows[i] + j] = adj_crows[vtx] + j;
            rowselect_vals[rowselect_crows[i] + j] = adj_vals[adj_crows[vtx] + j];
        }
    } 
}
void rowselect_csr_dupes_gpu(const at::Tensor& sampled_frontiers, const at::Tensor& adj_crows,
                            const at::Tensor& adj_cols, const at::Tensor& adj_vals,
                            const at::Tensor& rowselect_crows, const at::Tensor& rowselect_cols, 
                            const at::Tensor& rowselect_vals, int nnz_col_count, int nnz_count) {

    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz_col_count / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (nnz_col_count == 0 || nnz_count == 0) {
        return;
    }

    RowSelectCsrDupes<<<BLOCK_COUNT, BLOCK_SIZE>>>(sampled_frontiers.data<long>(), 
                                                    adj_crows.data<long>(), 
                                                    adj_cols.data<long>(),
                                                    adj_vals.data<long>(),
                                                    rowselect_crows.data<long>(),
                                                    rowselect_cols.data<long>(),
                                                    rowselect_vals.data<long>(),
                                                    nnz_col_count);

    CHECK_ERROR("rowselect_csr_dupes")
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse COO Tensor GPU-only constructor");
    m.def("sparse_csr_tensor_gpu", &sparse_csr_tensor_gpu, "Sparse CSR Tensor GPU-only constructor");
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
    m.def("normalize_csr_gpu", &normalize_csr_gpu, "Normalize rows in a CSR matrix");
    m.def("sum_csrd_gpu", &sum_csrd_gpu, "Sum rows in a CSR matrix (double values)");
    m.def("sum_csri_gpu", &sum_csri_gpu, "Sum rows in a CSR matrix (int values)");
    m.def("shift_rowselect_gpu", &shift_rowselect_gpu, "Shift row selection output matrix col values");
    m.def("shift_colselect_gpu", &shift_colselect_gpu, "Shift col selection matrix row values");
    // m.def("scatterd_add_gpu", &scatterd_add_gpu, "Implementation of scatter_add_ for doubles");
    m.def("scatteri_add_gpu", &scatteri_add_gpu, "Implementation of scatter_add_ for ints");
    m.def("rowselect_coo_gpu", &rowselect_coo_gpu, "COO row selection for sparsity-aware spgemm");
    m.def("coogeam_gpu", &coogeam_gpu, "csrgeam2 wrapper for cusparse");
    m.def("rowselect_csr_gpu", &rowselect_csr_gpu, "CSR row selection for sparsity-aware spgemm");
    m.def("normalize_batch_gpu", &normalize_batch_gpu, "Normalize SpMM output");
    m.def("sort_dst_proc_gpu", &sort_dst_proc_gpu, "Sort vertices by destination process");
    m.def("nsparse_spgemm", &nsparse_spgemm, "CUDA SpGEMM from NSparse");
    m.def("rearrange_rows_gpu", &rearrange_rows_gpu, "Rearrange rows in in place of local spgemm");
    m.def("rearrangel_rows_gpu", &rearrangel_rows_gpu, "Rearrange rows in in place of local spgemm 64-bit");
    m.def("reduce_sum_gpu", &reduce_sum_gpu, "Sum for SAGE sa-spgemm reduction");
    m.def("shadow_colselect_gpu", &shadow_colselect_gpu, "ShaDow column selection");
    m.def("rowselect_csr_dupes_gpu", &rowselect_csr_dupes_gpu, "CSR row selection with duplicates");
}

#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char** argv) {

  if(argc < 3) {
      std::cout << "Please specify the number of data size in thousands and number of gpus per node";
      return 0;
  }

  int n;
  int ngpus;
  n = atoi(argv[1]);
  ngpus = atoi(argv[2]);
  n *= 1000;

  int rank, size;

  float** sendbuff = (float**)malloc(ngpus * sizeof(float*));
  float** recvbuff = (float**)malloc(ngpus * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*ngpus);

  cudaEvent_t start[ngpus];
  cudaEvent_t stop[ngpus];

  float *h_data = new float[n]();
  for (int i = 0; i < n; i++) {
    h_data[i] = 5.0f;
  }

  // Initialize send/receive buffers, streams, and timers
  for (int i = 0; i < ngpus; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(recvbuff + i, n * sizeof(float)));

    if (rank == 0 && i == 0) {
      CUDACHECK(cudaMalloc(sendbuff + i, n * sizeof(float)));
      CUDACHECK(cudaMemcpy(sendbuff[i], h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDACHECK(cudaMemset(recvbuff[i], 0, n * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));

    CUDACHECK(cudaEventCreate(&start[i]));
    CUDACHECK(cudaEventCreate(&stop[i]));
  }

  ncclUniqueId id;
  ncclComm_t comms[ngpus];

  // Generating NCCL unique ID at one process and broadcasting it to all
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }

  // Initialize NCCL communicator.
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < ngpus; i++) {
     CUDACHECK(cudaSetDevice(i));
     NCCLCHECK(ncclCommInitRank(comms+i, size*ngpus, id, rank*ngpus + i));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int j = 0; j < 1; j++) {
      // Call ncclBroadcast (ncclGroup* calls make this function as one ncclBroadcast call).
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < ngpus; i++) {
        CUDACHECK(cudaEventRecord(start[i], s[i]));
        NCCLCHECK(ncclBroadcast((const void*)sendbuff[0], (void*)recvbuff[i], n, ncclFloat, 0,
            comms[i], s[i]));
      }
      NCCLCHECK(ncclGroupEnd());
  }

  // Synchronizing on CUDA stream to complete NCCL communication.
  for (int i = 0; i < ngpus; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
    CUDACHECK(cudaEventRecord(stop[i], s[i]));
    CUDACHECK(cudaEventSynchronize(stop[i]));
  }

  // Collect timings and verify broadcast worked.
  float *h_recvbuff = new float[n]();
  for (int i = 0; i < ngpus; i++) {
    CUDACHECK(cudaSetDevice(i));

    float time;
    CUDACHECK(cudaEventElapsedTime(&time, start[i], stop[i]));
    time /= 1000; // seconds

    cudaMemcpy(h_recvbuff, recvbuff[i], n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int j = 0; j < n; j++) {
      if (h_recvbuff[i] != 5.0f) {
        std::cout << "bcast error" << std::endl;
      }
    }
    std::cout << "rank: " << rank << " gpu: " << i << " size: " << (n * sizeof(float)) << " time: " << time << " bw: " << ((n * sizeof(float)) / time) << std::endl;
  } 

  // Freeing device memory
  for (int i = 0; i < ngpus; i++) {
     if (rank == 0 && i == 0) {
       CUDACHECK(cudaFree(sendbuff[i]));
      }
     CUDACHECK(cudaFree(recvbuff[i]));
  }


  // Finalizing NCCL
  for (int i=0; i<ngpus; i++) {
     ncclCommDestroy(comms[i]);
  }

  return 0;
}

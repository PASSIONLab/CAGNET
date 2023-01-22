#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


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


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  int size = 1*1024*1024*1024;


  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }


  ncclUniqueId id;
  ncclComm_t comm;
  float **sendbuffs = new float*[nRanks - 1]();
  float *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));

  if (myRank == 0) {
    for (int i = 0; i < nRanks - 1; i++) {
      CUDACHECK(cudaMalloc(&sendbuffs[i], size * sizeof(float)));
    }
  }
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));


  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  
  int warmup_runs = 1;
  cudaEvent_t start, stop;
  float total_time = 0.0f;
  for (int i = 0; i < warmup_runs + 1; i++) {
    if (myRank == 0) {
      float *h_sendbuff = new float[size]();
      for (int j = 0; j < size; j++) {
        h_sendbuff[j] = 2;
      }
      for (int j = 0; j < nRanks - 1; j++) {
        CUDACHECK(cudaMemcpy(sendbuffs[j], h_sendbuff, size * sizeof(float), cudaMemcpyHostToDevice));
      }
      if (i == warmup_runs) {
        CUDACHECK(cudaEventCreate(&start));
        CUDACHECK(cudaEventCreate(&stop));

        cudaEventRecord(start);
      }

      ncclGroupStart();
      for (int j = 0; j < nRanks - 1; j++) {
        NCCLCHECK(ncclSend((const void*)sendbuffs[j], size, ncclFloat, j + 1, comm, s));
      }
      ncclGroupEnd();
      //communicating using NCCL
      // NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
      //       comm, s));

      if (i == warmup_runs) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&total_time, start, stop);
      }
    } else {
      CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
      NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, 0, comm, s));
    }

  }

  if (myRank == 0) {
    total_time = total_time / 1000; // seconds
    float gb_count = (size * sizeof(float)) / ((float)(1 << 30));
    float bw = gb_count / total_time;
    std::cout << "total_time: " << total_time << std::endl;
    std::cout << "gb count: " << gb_count << " GB" << std::endl;
    std::cout << "bw: " << bw << " GB/s" << std::endl;
  } else {
    float *h_recvbuff = new float[size]();
    CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
      std::cout << "i: " << i << " val: " << h_recvbuff[i] << std::endl;
    }
  }


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));


  //free device buffers
  for (int i = 0; i < nRanks - 1; i++) {
    CUDACHECK(cudaFree(sendbuffs[i]));
  }
  delete[] sendbuffs;
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
// #include <iostream>
// #include <stdio.h>
// #include "cuda_runtime.h"
// #include "nccl.h"
// #include <unistd.h>
// #include <stdint.h>
// 
// #define CUDACHECK(cmd) do {                         \
//   cudaError_t e = cmd;                              \
//   if( e != cudaSuccess ) {                          \
//     printf("Failed: Cuda error %s:%d '%s'\n",             \
//         __FILE__,__LINE__,cudaGetErrorString(e));   \
//     exit(EXIT_FAILURE);                             \
//   }                                                 \
// } while(0)
// 
// 
// #define NCCLCHECK(cmd) do {                         \
//   ncclResult_t r = cmd;                             \
//   if (r!= ncclSuccess) {                            \
//     printf("Failed, NCCL error %s:%d '%s'\n",             \
//         __FILE__,__LINE__,ncclGetErrorString(r));   \
//     exit(EXIT_FAILURE);                             \
//   }                                                 \
// } while(0)
// 
// int main(int argc, char** argv) {
// 
//   if(argc < 3) {
//       std::cout << "Please specify the number of data size in thousands and number of gpus per node";
//       return 0;
//   }
// 
//   int n;
//   int ngpus;
//   n = atoi(argv[1]);
//   ngpus = atoi(argv[2]);
//   n *= 1000;
// 
//   int rank, size;
// 
//   float** sendbuff = (float**)malloc(ngpus * sizeof(float*));
//   float** recvbuff = (float**)malloc(ngpus * sizeof(float*));
//   cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*ngpus);
// 
//   cudaEvent_t start[ngpus];
//   cudaEvent_t stop[ngpus];
// 
//   float *h_data = new float[n]();
//   for (int i = 0; i < n; i++) {
//     h_data[i] = 5.0f;
//   }
// 
//   // Initialize send/receive buffers, streams, and timers
//   for (int i = 0; i < ngpus; i++) {
//     CUDACHECK(cudaSetDevice(i));
//     CUDACHECK(cudaMalloc(recvbuff + i, n * sizeof(float)));
// 
//     if (rank == 0 && i == 0) {
//       CUDACHECK(cudaMalloc(sendbuff + i, n * sizeof(float)));
//       CUDACHECK(cudaMemcpy(sendbuff[i], h_data, n * sizeof(float), cudaMemcpyHostToDevice));
//     }
// 
//     CUDACHECK(cudaMemset(recvbuff[i], 0, n * sizeof(float)));
//     CUDACHECK(cudaStreamCreate(s+i));
// 
//     CUDACHECK(cudaEventCreate(&start[i]));
//     CUDACHECK(cudaEventCreate(&stop[i]));
//   }
// 
//   ncclUniqueId id;
//   ncclComm_t comms[ngpus];
// 
//   // Generating NCCL unique ID at one process and broadcasting it to all
//   if (rank == 0) {
//     ncclGetUniqueId(&id);
//   }
// 
//   // Initialize NCCL communicator.
//   NCCLCHECK(ncclGroupStart());
//   for (int i = 0; i < ngpus; i++) {
//      CUDACHECK(cudaSetDevice(i));
//      NCCLCHECK(ncclCommInitRank(comms+i, size*ngpus, id, rank*ngpus + i));
//   }
//   NCCLCHECK(ncclGroupEnd());
// 
//   for (int j = 0; j < 1; j++) {
//       // Call ncclBroadcast (ncclGroup* calls make this function as one ncclBroadcast call).
//       NCCLCHECK(ncclGroupStart());
//       for (int i = 0; i < ngpus; i++) {
//         CUDACHECK(cudaEventRecord(start[i], s[i]));
//         NCCLCHECK(ncclBroadcast((const void*)sendbuff[0], (void*)recvbuff[i], n, ncclFloat, 0,
//             comms[i], s[i]));
//       }
//       NCCLCHECK(ncclGroupEnd());
//   }
// 
//   // Synchronizing on CUDA stream to complete NCCL communication.
//   for (int i = 0; i < ngpus; i++) {
//     CUDACHECK(cudaSetDevice(i));
//     CUDACHECK(cudaStreamSynchronize(s[i]));
//     CUDACHECK(cudaEventRecord(stop[i], s[i]));
//     CUDACHECK(cudaEventSynchronize(stop[i]));
//   }
// 
//   // Collect timings and verify broadcast worked.
//   float *h_recvbuff = new float[n]();
//   for (int i = 0; i < ngpus; i++) {
//     CUDACHECK(cudaSetDevice(i));
// 
//     float time;
//     CUDACHECK(cudaEventElapsedTime(&time, start[i], stop[i]));
//     time /= 1000; // seconds
// 
//     cudaMemcpy(h_recvbuff, recvbuff[i], n * sizeof(float), cudaMemcpyDeviceToHost);
//     for (int j = 0; j < n; j++) {
//       if (h_recvbuff[i] != 5.0f) {
//         std::cout << "bcast error" << std::endl;
//       }
//     }
//     std::cout << "rank: " << rank << " gpu: " << i << " size: " << (n * sizeof(float)) << " time: " << time << " bw: " << ((n * sizeof(float)) / time) << std::endl;
//   } 
// 
//   // Freeing device memory
//   for (int i = 0; i < ngpus; i++) {
//      if (rank == 0 && i == 0) {
//        CUDACHECK(cudaFree(sendbuff[i]));
//       }
//      CUDACHECK(cudaFree(recvbuff[i]));
//   }
// 
// 
//   // Finalizing NCCL
//   for (int i=0; i<ngpus; i++) {
//      ncclCommDestroy(comms[i]);
//   }
// 
//   return 0;
// }

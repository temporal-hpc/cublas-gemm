/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <cblas.h>
#include <omp.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define LIM_CHECK_N 4096
#define LIM_PRINT_N 32

// fraction error   1.0 is 100% 
#define TOLERR 0.0001

#include "tools.h"

using namespace std;

int main(int argc, char **argv) {
  cublasStatus_t status;
  if(argc != 5){
      fprintf(stderr, "run as ./prog dev nt n comptype\n\n");
      printArgsInfo();
      return EXIT_FAILURE;
  }
  int dev = atoi(argv[1]);
  int nt = atoi(argv[2]);
  int N = atoi(argv[3]);
  int comptype = atoi(argv[4]);
  // host pointers
  ATYPE *h_A;
  float *cblasA;
  BTYPE *h_B;
  float *cblasB;
  CTYPE *h_C;
  float *cblasC;
  // device pointers
  ATYPE *d_A = 0;
  BTYPE *d_B = 0;
  CTYPE *d_C = 0;
  // constants
  CTYPE alpha = 1.0f;
  CTYPE beta = 0.0f;
  // number of elements
  unsigned long nelem = N * N;
  double t1, t2;
  double TFLOP = 2.0*N*N*N * 1E-12;
  int bitsA = sizeof(ATYPE)*8;
  int bitsB = sizeof(BTYPE)*8;
  int bitsC = sizeof(CTYPE)*8;

  cudaDataType dtypeA = dataTypes[hmap(bitsA)];
  cudaDataType dtypeB = dataTypes[hmap(bitsB)];
  cudaDataType dtypeC = dataTypes[hmap(bitsC)];
  const char* dtypeAStr = dataTypesStr[hmap(bitsA)];
  const char* dtypeBStr = dataTypesStr[hmap(bitsB)];
  const char* dtypeCStr = dataTypesStr[hmap(bitsC)];

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasHandle_t handle;
  cudaSetDevice(dev);
  omp_set_num_threads(nt);
  printf("MATMUL A x B = C (%i x %i)\nA FP%i (%s)\nB FP%i (%s)\nC FP%i (%s)\n\n", 
          N, N,  
          bitsA, dtypeAStr,
          bitsB, dtypeBStr,
          bitsC, dtypeCStr);



  /* 1) Initialize CUBLAS */
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }


  /* 2) Set math mode */
  printf("Compute Type.....................%s\n", cublasComputeTypesStr[comptype]);
  //status = cublasSetMathMode(handle, cublasMathModes[0]);
  if (status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "!!!! CUBLAS MATH MODE ERROR\n");
    return EXIT_FAILURE;
  }


  /* 3) Allocate and fill host memory for the matrices */
  printf("Host mallocs A B C............."); fflush(stdout);
  t1 = omp_get_wtime();
  h_A = (ATYPE*)(malloc(nelem * sizeof(h_A[0])));
  h_B = (BTYPE*)(malloc(nelem * sizeof(h_B[0])));
  h_C = (CTYPE*)(malloc(nelem * sizeof(h_C[0])));
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);
  printf("Filling matrices in Host......."); fflush(stdout);
  t1 = omp_get_wtime();
  fillMatrixRand<ATYPE>(h_A, nelem);
  fillMatrixRand<BTYPE>(h_B, nelem);
  fillMatrixRand<CTYPE>(h_C, nelem);
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);
  print_matrix<ATYPE>(h_A, N, N, "MAT A");
  print_matrix<BTYPE>(h_B, N, N, "MAT B");


  /* 4) Allocate device memory for the matrices */
  printf("Device mallocs A B C..........."); fflush(stdout);
  t1 = omp_get_wtime();
  if (cudaMalloc(reinterpret_cast<void **>(&d_A), nelem * sizeof(d_A[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), nelem * sizeof(d_B[0])) != cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), nelem * sizeof(d_C[0])) != cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);



  /* 5) Initialize the device matrices with the host matrices */
  printf("Device -> Host memcpy A B C...."); fflush(stdout);
  t1 = omp_get_wtime();
  status = cublasSetVector(nelem, sizeof(h_A[0]), h_A, 1, d_A, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(nelem, sizeof(h_B[0]), h_B, 1, d_B, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(nelem, sizeof(h_C[0]), h_C, 1, d_C, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
    return EXIT_FAILURE;
  }
  t2 = omp_get_wtime();
  printf("done: %f secs\n\n", t2-t1); fflush(stdout);







  /* 6) GEMM -> GPU CUBLAS */
  printf("[CUBLAS] GPU GEMM.............."); fflush(stdout);
  cudaEventRecord(start);
  status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                    d_A, dtypeA, N,
                                    d_B, dtypeB, N,
                          &beta,    d_C, dtypeC, N, cublasComputeTypes[comptype], CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float gputime_ms;
  cudaEventElapsedTime(&gputime_ms, start, stop);
  double gpuTFLOPS = TFLOP/(gputime_ms/1000.0);
  printf("done: %f secs   [%f TFLOPS]\n", gputime_ms/1000.0, gpuTFLOPS); fflush(stdout);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }





  /* 7) GEMM -> CPU BASIC */
  //printf("[CBLAS] (float) Host mallocs A B C............."); fflush(stdout);
  t1 = omp_get_wtime();
  cblasA = (float*)(malloc(nelem * sizeof(cblasA[0])));
  cblasB = (float*)(malloc(nelem * sizeof(cblasB[0])));
  cblasC = (float*)(malloc(nelem * sizeof(cblasC[0])));
  t2 = omp_get_wtime();
  //printf("done: %f secs\n", t2-t1); fflush(stdout);
  //printf("[CBLAS] (float) Filling matrices in Host......."); fflush(stdout);
  t1 = omp_get_wtime();
  copyMatrix<float, ATYPE>(cblasA, h_A, N);
  copyMatrix<float, BTYPE>(cblasB, h_B, N);
  t2 = omp_get_wtime();
  //printf("done: %f secs\n", t2-t1); fflush(stdout);
  printf("[CBLAS] CPU GEMM..............."); fflush(stdout);
  t1 = omp_get_wtime();
  //cpuGemm(N, alpha, h_A, h_B, beta, h_C);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, cblasA, N, cblasB, N, beta, cblasC, N);
  t2 = omp_get_wtime();
  double cpuTFLOPS = TFLOP/(t2-t1);
  printf("done: %f secs   [%f TFLOPS]\n\n", t2-t1, cpuTFLOPS); fflush(stdout);
  print_matrix<float>(cblasC, N, N, "RESULT MAT C (CPU)");





  /* 8) Read the result back */
  printf("Device -> Host memcpy C........"); fflush(stdout);
  t1 = omp_get_wtime();
  status = cublasGetVector(nelem, sizeof(h_C[0]), d_C, 1, h_C, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);
  print_matrix<CTYPE>(h_C, N, N, "RESULT MAT C (GPU)");





  /* 9) Check result against reference */
  printf("Verify result.................."); fflush(stdout);
  t1 = omp_get_wtime();
  checkResult(cblasC, h_C, N, TOLERR); 
  t2 = omp_get_wtime();
  printf(": %f secs\n\n", t2-t1); fflush(stdout);






  /* 10) Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* 11) Shutdown */
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }
}

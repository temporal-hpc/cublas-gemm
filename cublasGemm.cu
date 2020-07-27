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

#define ATYPE float
#define BTYPE float
#define CTYPE float

#include "tools.h"

using namespace std;

int main(int argc, char **argv) {
  cublasStatus_t status;
  if(argc != 5){
      fprintf(stderr, "run as ./prog dev nt n mathMode\n\n");
      printArgsInfo();
      return EXIT_FAILURE;
  }
  int dev = atoi(argv[1]);
  int nt = atoi(argv[2]);
  int N = atoi(argv[3]);
  int mode = atoi(argv[4]);
  // host pointers
  ATYPE *h_A;
  BTYPE *h_B;
  CTYPE *h_C, *h_C_ref;
  // device pointers
  ATYPE *d_A = 0;
  BTYPE *d_B = 0;
  CTYPE *d_C = 0;
  // constants
  CTYPE alpha = 1.0f;
  CTYPE beta = 0.0f;
  // number of elements
  unsigned long nelem = N * N;
  float error_norm;
  float ref_norm;
  double t1, t2;
  double TFLOP = 2.0*N*N*N * 1E-12;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasHandle_t handle;
  cudaSetDevice(dev);
  omp_set_num_threads(nt);
  printf("CA Simulation (%i x %i)\n", N, N);



  /* 1) Initialize CUBLAS */
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }



  /* 2) Set math mode */
  printf("Math Mode......................%s\n", cublasMathModesStr[mode]);
  status = cublasSetMathMode(handle, cublasMathModes[mode]);
  if (status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "!!!! CUBLAS MATH MODE ERROR\n");
    return EXIT_FAILURE;
  }



  /* 3) Allocate and fill host memory for the matrices */
  h_A = (ATYPE*)(malloc(nelem * sizeof(h_A[0])));
  h_B = (BTYPE*)(malloc(nelem * sizeof(h_B[0])));
  h_C = (CTYPE*)(malloc(nelem * sizeof(h_C[0])));
  printf("Filling matrices in Host......."); fflush(stdout);
  t1 = omp_get_wtime();
  fillMatrixRand<ATYPE>(h_A, nelem);
  fillMatrixRand<BTYPE>(h_B, nelem);
  fillMatrixRand<CTYPE>(h_C, nelem);
  t2 = omp_get_wtime();
  printf("done: %f secs\n\n", t2-t1); fflush(stdout);
  print_matrix<ATYPE>(h_A, N, N, "MAT A");
  print_matrix<BTYPE>(h_B, N, N, "MAT B");


  /* 4) Allocate device memory for the matrices */
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



  /* 5) Initialize the device matrices with the host matrices */
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



  /* 6) GEMM -> CPU BASIC */
  printf("[CBLAS] CPU GEMM..............."); fflush(stdout);
  t1 = omp_get_wtime();
  //cpuGemm(N, alpha, h_A, h_B, beta, h_C);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, h_A, N, h_B, N, beta, h_C, N);
  t2 = omp_get_wtime();
  double cpuTFLOPS = TFLOP/(t2-t1);
  printf("done: %f secs   [%f TFLOPS]\n", t2-t1, cpuTFLOPS); fflush(stdout);
  print_matrix<CTYPE>(h_C, N, N, "RESULT MAT C (CPU)");
  h_C_ref = h_C;



  /* 7) GEMM -> GPU CUBLAS */
  printf("[CUBLAS] GPU GEMM.............."); fflush(stdout);
  cudaEventRecord(start);
  status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                          d_A, CUDA_R_16F, N,
                          d_B, CUDA_R_16F, N,
                          &beta, d_C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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



  /* 8) Allocate host memory for reading back the result from device memory */
  h_C = reinterpret_cast<CTYPE *>(malloc(nelem * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* 9) Read the result back */
  status = cublasGetVector(nelem, sizeof(h_C[0]), d_C, 1, h_C, 1);
  print_matrix<CTYPE>(h_C, N, N, "RESULT MAT C (GPU)");

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }

  /* 10) Check result against reference */
  checkResult(h_C_ref, h_C, N, nelem, &error_norm, &ref_norm); 


  /* 11) Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

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

  /* 12) Shutdown */
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }
}

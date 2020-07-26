/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <omp.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define ATYPE __half
#define BTYPE __half
#define CTYPE float

#include "tools.h"

int main(int argc, char **argv) {
  cublasStatus_t status;
  if(argc != 4){
      fprintf(stderr, "run as ./prog dev n mathMode\n\n");
      printArgsInfo();
      return EXIT_FAILURE;
  }
  int dev = atoi(argv[1]);
  int N = atoi(argv[2]);
  int mode = atoi(argv[3]);
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
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  double t1, t2;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasHandle_t handle;
  cudaSetDevice(dev);
  printf("CA Simulation (%i x %i)\n", N, N);



  /* 1) Initialize CUBLAS */
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }



  /* 2) Set the math mode */
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
  #pragma omp parallel for
  for (i = 0; i < nelem; i++) {
    h_A[i] = 0.001;
    h_B[i] = 1.0;
    h_C[i] = 0.001;
  }
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
  printf("[MANUAL] CPU GEMM.............."); fflush(stdout);
  t1 = omp_get_wtime();
  cpuGemm(N, alpha, h_A, h_B, beta, h_C);
  t2 = omp_get_wtime();
  printf("done: %f secs\n", t2-t1); fflush(stdout);
  print_matrix<CTYPE>(h_C, N, N, "RESULT MAT C (CPU)");
  h_C_ref = h_C;



  /* 7) GEMM -> GPU CUBLAS */
  printf("[CUBLAS] GPU GEMM.............."); fflush(stdout);
  cudaEventRecord(start);
  //status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
  status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                          d_A, CUDA_R_16F, N,
                          d_B, CUDA_R_16F, N,
                          &beta, d_C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float gputime_ms;
  cudaEventElapsedTime(&gputime_ms, start, stop);
  printf("done: %f secs   [%f TFLOPS]\n", gputime_ms/1000.0, 
          ((double)N*N*(2*N+3))/((gputime_ms/1000.0)*1000*1000*1000.0*1000.0)); fflush(stdout);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }



  /* Allocate host memory for reading back the result from device memory */
  h_C = reinterpret_cast<CTYPE *>(malloc(nelem * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Read the result back */
  status = cublasGetVector(nelem, sizeof(h_C[0]), d_C, 1, h_C, 1);
  print_matrix<CTYPE>(h_C, N, N, "RESULT MAT C (GPU)");

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }

  /* Check result against reference */
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < nelem; ++i) {
    diff = (float)h_C_ref[i] - (float)h_C[i];
    error_norm += diff * diff;
    ref_norm += (float)h_C_ref[i] * (float)h_C_ref[i];
  }

  error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
  ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));

  if (fabs(ref_norm) < 1e-7) {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }

  /* Memory clean up */
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

  /* Shutdown */
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  if (error_norm / ref_norm < 1e-6f) {
    printf("CUBLAS test passed.\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("CUBLAS test failed.\n");
    exit(EXIT_FAILURE);
  }
}

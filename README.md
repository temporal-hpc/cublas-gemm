# CUBLAS GEMM Minimal Example

A simple CUBLAS v11.0 GEMM Example to start doing accelerated Linear Algebra on GPUs.

## 1) Versions history:
	- Version 1.0 (July 2020)

## 2) Requirements:
	- Nvidia GPU supporting CUDA
	- CUDA v11.0 or greater
    - CUBLAS v11.0 (should come with CUDA)
    - openblas (max-perf CPU test)

## 3) License:
    None yet. Use it at free will.

## 4) Install:
	1) Clone Repo:
        foo@bar:~$ git clone https://github.com/temporal-hpc/cublas-gemm
	2) Compile:
        cd cublas-gemm
        make
	3) Run:
        ./prog dev n nt computeType
        dev: GPU ID
        n: matrix of n x n
        nt: number of CPU threads (for CBLAS and data init)
        computeType: GPU Compute type (16F, 32F, 64F)
    4) Compile Options:
        You can specify the data type (half, float) for each matrix
        Example:
        make ATYPE=half BTYPE=half CTYPE=hal
        
## 5) Examples:
    a) Multiply 4096 x 4096 matrices using default CUBLAS math (FP32)
        make ATYPE=float BTYPE=float CTYPE=float
        ./prog 0 4 $((1024*4)) 2

    b) Multiply 4096 x 4096 matrices using Tensor Cores with mixed precision
        make ATYPE=half BTYPE=half CTYPE=float
        ./prog 0 4 $((1024*4)) 4

    c) Multiply 4096 x 4096 matrices using Tensor Cores full FP16
        make ATYPE=half BTYPE=half CTYPE=half
        ./prog 0 4 $((1024*4)) 0

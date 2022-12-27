# CUBLAS GEMM Minimal Example v1.1

A simple CUBLAS v11.0 GEMM Example to start doing accelerated Linear Algebra on GPUs.

## 1) Versions history:
    - v1.1 (September 2020)
	- v1.0 (July 2020)

## 2) Requirements:
	- Nvidia GPU supporting CUDA
	- CUDA v11.0 or greater
	- CUBLAS v11.0 (should come with CUDA)
	- openblas (max-perf CPU test)

## 3) License:
    None yet. Use it at free will.

## 4) Install and Compile:
	a) Clone Repo:
        git clone https://github.com/temporal-hpc/cublas-gemm
	b) Compile:
        cd cublas-gemm
        make

## 5) Run:
    a) Run:
        run as ./prog dev nt n comptype mode

        dev:      Device ID
        nt:       Number of CPU threads (accelerates data init and CPU mode)
        n:        Matrix size of n x n
        comptype: GPU CUBLAS mode
        mode:     CPU=0,  GPU=1

    b) CUBLAS Compute Types:
            0  = CUBLAS_COMPUTE_16F
            1  = CUBLAS_COMPUTE_16F_PEDANTIC
            2  = CUBLAS_COMPUTE_32F
            3  = CUBLAS_COMPUTE_32F_PEDANTIC
            4  = CUBLAS_COMPUTE_32F_FAST_16F
            5  = CUBLAS_COMPUTE_32F_FAST_16BF
            6  = CUBLAS_COMPUTE_32F_FAST_TF32
            7  = CUBLAS_COMPUTE_64F
            8  = CUBLAS_COMPUTE_64F_PEDANTIC
            9  = CUBLAS_COMPUTE_32I
            10 = CUBLAS_COMPUTE_32I_PEDANTIC

    c) Compile Options:
        You can specify the data type (half, float) for each matrix
        Example:
        make ATYPE=half BTYPE=half CTYPE=hal
        
## 6) Example executions:
    a) Multiply 4096 x 4096 matrices using default CUBLAS math (FP32)
        make ATYPE=float BTYPE=float CTYPE=float
        ./prog 0 4 $((1024*4)) 2

    b) Multiply 4096 x 4096 matrices using Tensor Cores with mixed precision
        make ATYPE=half BTYPE=half CTYPE=float
        ./prog 0 4 $((1024*4)) 4

    c) Multiply 4096 x 4096 matrices using Tensor Cores full FP16
        make ATYPE=half BTYPE=half CTYPE=half
        ./prog 0 4 $((1024*4)) 0

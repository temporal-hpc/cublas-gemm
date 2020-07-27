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
        ```console
        foo@bar:~$ git clone https://github.com/temporal-hpc/cublas-gemm
        ```
	2) Compile:
        ```console
        foo@bar:~$ cd cublas-gemm
        foo@bar:~$ make
        ```
	3) Run (Run as ./prog for info on args):
        ```
            ./prog n mode
        ```

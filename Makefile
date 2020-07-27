all: 
	nvcc -O3 cublasGemm.cu -lcublas -lcblas -Xcompiler -fopenmp -o prog

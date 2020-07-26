all: 
	nvcc cublasGemm.cu -l cublas -Xcompiler -fopenmp -o prog

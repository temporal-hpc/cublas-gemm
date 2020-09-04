INCDIR=
LIBDIR=
all: 
	nvcc -O3 -I${INCDIR} -L${LIBDIR} main.cu -lcublas -lcblas -Xcompiler -fopenmp -o prog

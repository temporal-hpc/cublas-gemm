INCDIR=
LIBDIR=
ARCH=sm_70
# possible types: float | half
ATYPE=float
BTYPE=float
CTYPE=float
all: 
	nvcc -arch=${ARCH} -O3 -I${INCDIR} -L${LIBDIR} -DATYPE=${ATYPE} -DBTYPE=${BTYPE} -DCTYPE=${CTYPE} main.cu -lcublas -lcblas -Xcompiler -fopenmp -o prog

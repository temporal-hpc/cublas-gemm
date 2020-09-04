INCDIR=
LIBDIR=
# possible types: float | half
ATYPE=float
BTYPE=float
CTYPE=float
all: 
	nvcc -O3 -I${INCDIR} -L${LIBDIR} -DATYPE=${ATYPE} -DBTYPE=${BTYPE} -DCTYPE=${CTYPE} main.cu -lcublas -lcblas -Xcompiler -fopenmp -o prog

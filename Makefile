INCDIR=
LIBDIR=
ARCH=sm_70
# possible types: half | float | double
ATYPE=float
BTYPE=float
CTYPE=float
PINNED=no
INCS=-I${INCDIR} 
LIBS=-L${LIBDIR} 
DEFS=-DATYPE=${ATYPE} -DBTYPE=${BTYPE} -DCTYPE=${CTYPE} -D${PINNED} 
CUDAOPTS=-arch=${ARCH} -O3 -lcublas -lcblas -lopenblas -Xcompiler -fopenmp 
all: 
	nvcc ${CUDAOPTS} ${INCS} ${LIBS} ${DEFS} main.cu -o prog

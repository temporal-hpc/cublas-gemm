INCDIR=
LIBDIR=
ARCH=sm_70
# possible types: half | float | double
ATYPE=float
BTYPE=float
CTYPE=float
CPUFP64=noCPUFP64
PINNED=noPINNED
INCS=-I${INCDIR} 
LIBS=-L${LIBDIR} 
DEFS=-DATYPE=${ATYPE} -DBTYPE=${BTYPE} -DCTYPE=${CTYPE} -D${CPUFP64} -D${PINNED} 
CUDAOPTS=-arch=${ARCH} -O3 -lcublas -lopenblas -Xcompiler -fopenmp 
all: 
	nvcc ${CUDAOPTS} ${INCS} ${LIBS} ${DEFS} main.cu -o prog

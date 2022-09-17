# ------------------------------------------------------------------------------
#  Makefile 
# ------------------------------------------------------------------------------
ALLOBJS = util.o lsh.o bucket.o bin.o assign.o eval.o seeding.o main.o

OPENMP  = -Xcompiler -fopenmp
OPT     = -w -O3
ARCH    = -gencode arch=compute_61,code=sm_61
CFLAGS  = -I/usr/mpi/gcc/openmpi-1.10.4-hfi/include
LDFLAGS = -L/usr/mpi/gcc/openmpi-1.10.4-hfi/lib64
FLAGS   = -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm

# ------------------------------------------------------------------------------
#  Compiler with OpenMP (first .cu, then .cc, order-sensitive)
# ------------------------------------------------------------------------------
all:$(ALLOBJS)
	nvcc -std=c++11 $(OPENMP) $(OPT) -o silk $(ARCH) -lm -lcudart -lcublas \
		$(CFLAGS) $(LDFLAGS) $(FLAGS) $(ALLOBJS)

%.o: %.cu
	nvcc -std=c++11 $(OPENMP) -c $(OPT) $(ARCH) $(CFLAGS) -o $@ $<

%.o: %.cc
	nvcc -std=c++11 $(OPENMP) -c $(OPT) $(ARCH) $(CFLAGS) -o $@ $<

clean:
	-rm $(ALLOBJS) silk

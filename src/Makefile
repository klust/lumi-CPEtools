# Makefile for the Tutorial examples.
# The Makefile should be compatible with EasyBuild when compiled without 
# MPI and OpenMP enabled, tough you will need to define OMPFLAG as the option
# that should be passed to the compiler to compile with OpenMP.
#
# IMPORTANT: Run with make -R (or make --no-builtin-variables) if CC etc are 
# not defined in the environment or the make command line as otherwise the 
# built-in values will be used rather than the defaults given below.
#
# Note: Preprocessor macros that can be defined via DEFINES
#  * DEBUG: Prints some debug information in hybrid_hello.
#  * HAVE_NUMALIB: Define this if the system has libnuma (relevant for
#    hybrid_hello)
#

CC      ?= cc
MPICC   ?= cc
CFLAGS  ?= -O2
OMPFLAG ?= -fopenmp
DEFINES ?= 
#DEFINES ?= -DDEBUG
LIBS    ?= 

.SUFFIXES:

examples : bindir \
	       ../bin/serial_check ../bin/omp_check ../bin/mpi_check ../bin/hybrid_check \
	       ../bin/xldd

clean : 
	-/bin/rm ../bin/*
	
bindir :
	-mkdir -p ../bin
	
#
# Cray acheck-like programs, common source to not have
# to maintain multiple very similar versions.
#

serial_check : ../bin/serial_check
../bin/serial_check : hybrid_check.c Makefile
	$(CC) $(CFLAGS) $(DEFINES) $(LIBS) $< -o $@

omp_check : ../bin/omp_check
../bin/omp_check : hybrid_check.c Makefile
	$(MPICC) $(CFLAGS) $(OMPFLAG) $(DEFINES) -DWITH_OMP $(LIBS) $< -o $@

mpi_check : ../bin/mpi_check
../bin/mpi_check : hybrid_check.c Makefile
	$(MPICC) $(CFLAGS) $(OMPFLAG) $(DEFINES) -DWITH_MPI $(LIBS) $< -o $@

hybrid_check : ../bin/hybrid_check
../bin/hybrid_check : hybrid_check.c Makefile
	$(MPICC) $(CFLAGS) $(OMPFLAG) $(DEFINES) -DWITH_MPI -DWITH_OMP $(LIBS) $< -o $@
	
../bin/xldd : xldd
	cp $< $@

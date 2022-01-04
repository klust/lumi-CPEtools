ml calcua/2020a intel/2020a

pushd ../../src
make CC=icc MPICC=mpiicc CFLAGS="-O2" OMPFLAG="-qopenmp" DEFINES="-DHAVE_NUMALIB" LIBS="-lnuma"
popd

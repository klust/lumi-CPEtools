#! /bin/bash
#SBATCH --time=0:20
#SBATCH --partition=debug
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --hint=nomultithread

#ml LUMI/21.08 partition/L
#ml cpeGNU

env | egrep ^SLURM

env | egrep ^OMP_

# It looks like when we start without srun, setting the right number of threads and setting OMP_PLACES
# is essential to ensure that only one virtual core per core is used.
# * --hint=nomultithread has no effect unless srun is used.
#
# cpeGNU:
# * OMP_NUM_THREADS is essential to get the right number of threads. Otherwise two threads are started
#   for each allocated CPU (with cpeGNU)
# * OMP_PLACES=cores sets CPU masks to {0,12}, {1, 129}, etc., i.e., all virtual cores for a core.
# * PROC_BIND=true seems to be automatically implied when using OMP_PLACES.
../../../bin/serial_check -l test -w 5 -r

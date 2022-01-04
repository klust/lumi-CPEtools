#! /bin/bash
#SBATCH --time=0:20
#SBATCH --partition=debug
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --hint=nomultithread

#ml LUMI/21.08 partition/L
#ml cpeGNU

env | egrep ^SLURM

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
#export OMP_PLACES=cores

env | egrep ^OMP_

# Using srun ensures that --hint=nomultithread is honored, but the threads are in no particular order
# without using OMP_ environment variables.
# Adding OMP_PROC_BIND=true however ensures that threads are nicely bound to cores and in order of the
# core numbers.
srun ../../../bin/omp_check -l test -w 5 -r

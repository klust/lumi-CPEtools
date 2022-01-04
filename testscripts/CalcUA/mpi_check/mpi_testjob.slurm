#! /bin/bash
#SBATCH --time=0:20
#SBATCH --partition=debug
#SBATCH --ntasks=8 --cpus-per-task=1

module --force purge
ml calcua/2020a
ml intel/2020a

export OMP_NUM_THREADS=1

srun ../../../bin/mpi_check -l test -w 5 -n -r

#! /bin/bash
#SBATCH --time=0:20
#SBATCH --partition=debug
#SBATCH --ntasks=256 --cpus-per-task=1

#ml LUMI/21.08 partition/L
#ml cpeGNU

srun ../../../bin/mpi_check -l test -w 5 -n -r

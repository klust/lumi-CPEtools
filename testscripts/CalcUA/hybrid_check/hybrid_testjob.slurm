#! /bin/bash
#SBATCH --time=2:00
#SBATCH --partition=debug
#SBATCH --ntasks=4 --cpus-per-task=4

module --force purge
ml calcua/2020a
ml intel/2020a

export OMP_PROC_BIND=true
srun ../../../bin/hybrid_check -l test -w 5 -n -r

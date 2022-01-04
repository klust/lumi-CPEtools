#! /bin/bash
#SBATCH --time=0:20
#SBATCH --partition=debug
#SBATCH --ntasks=1 --cpus-per-task=8

module --force purge
ml calcua/2020a
ml intel/2020a

export OMP_PROC_BIND=true

../../../bin/hybrid_check -l test -w 5 -n -r

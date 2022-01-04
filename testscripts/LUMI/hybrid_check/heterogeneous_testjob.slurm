#! /bin/bash
#SBATCH --time=0:45
#SBATCH --partition=small
#SBATCH --ntasks=4 --cpus-per-task=16
#SBATCH hetjob
#SBATCH --partition=small
#SBATCH --ntasks=8 --cpus-per-task=8

#ml LUMI/21.08 partition/L
#ml cpeGNU

ml

export OMP_PROC_BIND=true

srun --ntasks=4 --cpus-per-task=16 ../../../bin/hybrid_check -l exe1 -w 5 -n -r : \
     --ntasks=8 --cpus-per-task=8  ../../../bin/hybrid_check -l exe2

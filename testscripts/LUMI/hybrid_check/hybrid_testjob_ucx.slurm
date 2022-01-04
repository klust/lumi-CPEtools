#! /bin/bash
#SBATCH --time=2:00
#SBATCH --partition=debug
#SBATCH --ntasks=18 --cpus-per-task=8

#ml LUMI/21.08 partition/L
#ml cpeGNU
ml -cray-mpich craype-network-ucx cray-mpich-ucx

ml

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
#export OMP_PLACES=cores

#srun ../../../bin/hybrid_check -l test -w 5 -n -r
srun ../../../bin/hybrid_check -l test

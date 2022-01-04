#! /bin/bash
#SBATCH --time=2:00
#SBATCH --partition=debug
#SBATCH --ntasks=2 --cpus-per-task=8

#ml LUMI/21.08 partition/L
#ml cpeGNU
##ml -cray-mpich craype-network-ucx cray-mpich-ucx
##ml -xpmem

ml

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
#export OMP_PLACES=cores

#srun ../../../bin/hybrid_check -l test -w 15 -n -r
srun ../../../bin/hybrid_check -l test -w 15

echo -e "\n\nJob info:\n"
sstat -j $SLURM_JOB_ID -a

#!/bin/bash
#SBATCH --job-name=hybrid_gpu_check-test
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --hint=nomultithread
#SBATCH --time=1:00
#SBATCH --output %x-%j.txt
## The following line does not work as it seems not to pass other variables that are 
## important to clean the environment.
###SBATCH --export=LUMI_STACK_VERSION,EBVERSIONLUMIMINCPETOOLS

echo "Submitted from $SLURM_SUBMIT_HOST"
echo "Running on $SLURM_JOB_NODELIST"

echo
egrep "^#SBATCH" $0

echo
env | egrep ^SLURM

module list

export MPICH_GPU_SUPPORT_ENABLED=0
export DEVICE_STACK_MEM_SIZE=2.0
export HSA_XNACK=1

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "OMP variables:"
env | grep -e ^OMP_

cpu_bind="--cpu-bind=mask_cpu:0xfefe,0xfefe0000,0xfefe00000000,0xfefe000000000000"
gpu_bind="--gpu-bind=mask_gpu:0x30,0x0c,0xc0,0x03"
	
echo "CPU bind argument: $cpu_bind"
echo "GPU binding: $gpu_bind"
	
echo -e "\nsrun -N ${SLURM_NNODES} -n 4 ${cpu_bind} ${gpu_bind} gpu_check --fl"
timeout -k 10s 20s srun -N ${SLURM_NNODES} -n 4 ${cpu_bind} ${gpu_bind} ../../../bin/gpu_check --fl

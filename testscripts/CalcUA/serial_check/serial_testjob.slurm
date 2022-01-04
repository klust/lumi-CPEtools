#! /bin/bash
#SBATCH --time=0:20
#SBATCH --partition=debug
#SBATCH --ntasks=1 --cpus-per-task=1

module --force purge
ml calcua/2020a
ml intel/2020a

../../../bin/serial_check -l test -w 5 -n -r

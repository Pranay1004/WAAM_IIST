#!/bin/bash
#SBATCH --job-name=tip   # create a short name for your job
#SBATCH --time=48:00:00 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o slurm.%N.%j.out #STDOUT
#SBATCH -e log.err#STDERR
#SBATCH --export=all

module load gnu12

# Change to the directory where the executable is located
cd /home/varad

# Run the executable with OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./tip1.
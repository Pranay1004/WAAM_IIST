#!/bin/bash
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=10     # total number of tasks per node
##SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -J pure
#SBATCH -p iist
#SBATCH --time=48:00:00 
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --export=all

cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST > hostfile_$SLURM_JOBID

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK



module load openmpi4
module load gnu12

mpirun -mca btl self,vader /home/varad/tip1. >& out_$SLURM_JOBID
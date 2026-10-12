#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=25     # total number of tasks per node
#SBATCH -J test_1
#SBATCH -p iist
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --export=all

cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST > hostfile_$SLURM_JOBID





module load openmpi4
module load gnu12

mpirun -mca btl self,vader /home/varad/tip1. >& out_$SLURM_JOBID
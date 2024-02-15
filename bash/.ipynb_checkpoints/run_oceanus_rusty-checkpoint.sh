#!/bin/bash 

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rbrooks@flatironinstitute.org
#SBATCH --time=00:05:00
#SBATCH --job-name=oceanus-loop
#SBATCH -N1 --ntasks-per-node=2
#SBATCH --array=0 -N1 --ntasks-per-node=2
#SBATCH -e stderr.txt
#SBATCH -o stdout.txt

module purge
module load slurm
module load gcc
module load gsl
module load openmpi/4.0.7
module load hdf5
module load python

echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

VENVDIR=/mnt/home/rbrooks/ceph/venvs
source $VENVDIR/mwlmc_venv/bin/activate

START_NUM=$(($SLURM_ARRAY_TASK_ID))

pipeline=/mnt/home/rbrooks/ceph/oceanus/src/streamgenerator.py
paramline=/mnt/home/rbrooks/ceph/oceanus/ics/param-files/
python3 $pipeline ${paramline}param_$START_NUM.yaml 
#!/bin/bash

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rbrooks@flatironinstitute.org
#SBATCH --time=00:30:00
#SBATCH --job-name=agama-singlestream
#SBATCH -N1 --ntasks-per-node=1
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

pipeline=/mnt/home/rbrooks/ceph/oceanus/src/streamgen-agama.py
param=/mnt/home/rbrooks/ceph/oceanus/ics/high-vel-dis/agama-mw/param_396.yaml
python3 $pipeline -o -f $param

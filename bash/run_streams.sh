#!/bin/bash

id=$1
shift
cd /mnt/home/rbrooks/ceph/oceanus/src/
module purge
module load slurm
module load gcc
module load gsl
module load openmpi/4.0.7
module load hdf5
module load python
module load disBatch 

VENVDIR=/mnt/home/rbrooks/ceph/venvs
source $VENVDIR/mwlmc_venv/bin/activate

mkdir -p ../logs/

python streamgenerator.py "$@" &> ../logs/disbatch_${id}.log

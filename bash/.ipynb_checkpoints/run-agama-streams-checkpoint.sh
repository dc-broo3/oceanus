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

#-o to overwrite after .py 
python streamgen-agama.py -o -f "$@" &> ../logs/disbatch_${id}.log

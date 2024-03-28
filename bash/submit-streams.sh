#!/bin/bash

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rbrooks@flatironinstitute.org
#SBATCH --job-name=oceanus

module load disBatch
# Change Tasks file as required.
sbatch -N1 --constraint=rome -p cca -e streams.e disBatch -p ./log-dbUtil-status-files/ Tasks-staticmw
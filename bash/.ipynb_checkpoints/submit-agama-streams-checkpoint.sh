#!/bin/bash
module load disBatch
# Change Tasks file as required.
# sbatch -N8 --constraint=rome -p cca -e streams.e disBatch --mailFreq 1024 --mailTo rbrooks@flatironinstitute.org -p ./log-dbUtil-status-files/ --no-retire tasks/Tasks-agama 
sbatch -N8 -t 01:00:00 --constraint=rome -p cca -e streams.e disBatch --mailFreq 1024 --mailTo rbrooks@flatironinstitute.org -p ./log-dbUtil-status-files/ --no-retire tasks/Tasks-agama 
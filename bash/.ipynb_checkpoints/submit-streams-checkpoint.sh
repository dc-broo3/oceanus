#!/bin/bash
module load disBatch
# Change Tasks file as required.
sbatch -N16 --constraint=rome -p cca -e streams.e disBatch --mailFreq 16384 --mailTo rbrooks@flatironinstitute.org -p ./log-dbUtil-status-files/ --no-retire tasks/Tasks-rm-mwh
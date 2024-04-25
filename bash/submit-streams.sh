#!/bin/bash

module load disBatch
# Change Tasks file as required.
sbatch -N8 --constraint=rome -p cca -e streams.e disBatch --mailFreq 1024 --mailTo rbrooks@flatironinstitute.org -p ./log-dbUtil-status-files/ Tasks-full-mwh
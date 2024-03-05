#!/bin/bash
module load disBatch
#change Tasks to which potential we want 
sbatch -N8 --constraint=rome -p cca -e streams.e disBatch -p ./log-dbUtil-status-files/ Tasks-em-mwh


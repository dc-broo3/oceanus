#!/bin/bash
module load disBatch
sbatch -N1 --constraint=rome -p cca -e streams.e disBatch -p ./log-dbUtil-status-files/ Tasks-128streams


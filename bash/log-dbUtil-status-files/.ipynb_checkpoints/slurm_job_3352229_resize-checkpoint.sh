export SLURM_NODELIST="worker[5334,5339-5340]"
export SLURM_JOB_NODELIST="worker[5334,5339-5340]"
export SLURM_NNODES=3
export SLURM_JOB_NUM_NODES=3
export SLURM_JOB_CPUS_PER_NODE="128(x3)"
unset SLURM_NPROCS
unset SLURM_NTASKS
unset SLURM_TASKS_PER_NODE

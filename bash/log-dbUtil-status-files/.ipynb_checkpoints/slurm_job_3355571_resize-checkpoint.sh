export SLURM_NODELIST="worker5302"
export SLURM_JOB_NODELIST="worker5302"
export SLURM_NNODES=1
export SLURM_JOB_NUM_NODES=1
export SLURM_JOB_CPUS_PER_NODE="128"
unset SLURM_NPROCS
unset SLURM_NTASKS
unset SLURM_TASKS_PER_NODE
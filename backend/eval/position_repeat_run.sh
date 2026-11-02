#!/bin/bash
# eval/position_repeat.py on the cluster: MODE=probs (one GPU job) or MODE=score (CPU array on courses).
#   sbatch ... --export=ALL,MODE=probs eval/position_repeat_run.sh 1
#   sbatch ... --array=0-39 --export=ALL,MODE=score eval/position_repeat_run.sh 40
N=$1
module load miniconda3/25.9.1; source activate ~/envs/arr
export NUMBA_CACHE_DIR=/tmp/numba_$SLURM_JOB_ID   # node-local: a cache compiled on another CPU segfaulted
export PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval
P=/scratch/basu.ang/position_repeat/frozen.py
if [ "$MODE" = "probs" ]; then
    python $P --probs 2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"
else
    python $P --score --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards $N 2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"
fi

#!/bin/bash
# CPU half of eval/never_kept.py (courses): scores the songs the GPU half has finished.
#   sbatch --array=0-39 eval/never_kept_score.sh 40
N=$1
module load miniconda3/25.9.1; source activate ~/envs/arr
cd /scratch/basu.ang/never_kept
PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval \
    python /scratch/basu.ang/never_kept/frozen.py --score --songs songs.json --out /scratch/basu.ang/never_kept \
    --shard $SLURM_ARRAY_TASK_ID --nshards $N 2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"

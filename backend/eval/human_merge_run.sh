#!/bin/bash
# CPU job for eval/human_merge.py (courses):   sbatch --array=0-39 eval/human_merge_run.sh 40
N=$1
module load miniconda3/25.9.1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval \
    python /scratch/basu.ang/human_merge/frozen.py --score $EXTRA --shard $SLURM_ARRAY_TASK_ID --nshards $N 2>&1 \
    | grep --line-buffered -v -i "warn\|pkg_resources\|nanmean\|return np"

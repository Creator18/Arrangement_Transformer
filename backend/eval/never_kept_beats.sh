#!/bin/bash
# CPU half before the GPU half of eval/never_kept.py (courses / short): madmom beats for every song still to run,
# cached in /scratch/basu.ang/never_kept/beats.  The GPU half then reads them (never_kept_run.sh, --cached-only).
#   sbatch -A cs7170.202710 -p courses -c 1 --mem=6G --time=04:00:00 --array=0-39 eval/never_kept_beats.sh 40
N=$1
module load miniconda3/25.9.1; source activate ~/envs/arr
cd /scratch/basu.ang/never_kept
PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval \
    python /scratch/basu.ang/never_kept/frozen.py --beats --songs songs.json --out /scratch/basu.ang/never_kept \
    --shard $SLURM_ARRAY_TASK_ID --nshards $N 2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"

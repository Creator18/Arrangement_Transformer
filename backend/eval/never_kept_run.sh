#!/bin/bash
# GPU half of eval/never_kept.py: the hosted chain on every Music2MIDI song (skips songs already done, so a piece that
# times out is simply submitted again).  Array task k of n = shard k.  Only songs whose madmom beats
# never_kept_beats.sh has cached: madmom inside this job left the GPU idle ~97% of the time and the cluster cancelled it.
#   sbatch --array=0-11 eval/never_kept_run.sh 12
N=$1
module load miniconda3/25.9.1; source activate ~/envs/arr
cd /scratch/basu.ang/never_kept
echo "$(date) shard $SLURM_ARRAY_TASK_ID/$N on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval \
    python /scratch/basu.ang/never_kept/frozen.py --run --songs songs.json --out /scratch/basu.ang/never_kept \
    --shard $SLURM_ARRAY_TASK_ID --nshards $N --cached-only 2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"

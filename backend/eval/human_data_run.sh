#!/bin/bash
# eval/human_data.py on the GPU: the base's raw notes + clean-up features + human labels for every Music2MIDI song.
# Skips songs already done, so a piece that stops is simply submitted again.   sbatch --array=0-7 ... 8
N=$1
module load miniconda3/25.9.1; source activate ~/envs/arr
echo "$(date) shard $SLURM_ARRAY_TASK_ID/$N on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval \
    python ~/arrangement_transformer/backend/eval/human_data.py --songs /scratch/basu.ang/never_kept/songs.json \
    --nk /scratch/basu.ang/never_kept --out /scratch/basu.ang/human_data --shard $SLURM_ARRAY_TASK_ID --nshards $N \
    2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"

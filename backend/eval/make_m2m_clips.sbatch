#!/bin/bash
# Music2MIDI -> spectral song-to-piano clips, one shard per array task.
#   sbatch --array=0-15 ~/arrangement_transformer/backend/eval/make_m2m_clips.sbatch
#SBATCH -J m2m-clips
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/m2m_clips_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
M=/scratch/basu.ang/m2m
python -u eval/make_m2m_clips.py --out $M/clips --audio-dir $M/audio --data-dir $M \
    --splits $M/m2m_splits.json --shard ${SLURM_ARRAY_TASK_ID}/16

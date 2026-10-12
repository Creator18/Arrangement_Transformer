#!/bin/bash
# Arrangement scorecard over Music2MIDI human covers, one shard per array task.
#   sbatch --array=0-31 ~/arrangement_transformer/backend/eval/scorecard.sbatch
#SBATCH -J scorecard
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/scorecard_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
python -u eval/scorecard.py --m2m /scratch/basu.ang/m2m --out /scratch/basu.ang/m2m/scorecard \
    --shard ${SLURM_ARRAY_TASK_ID}/32

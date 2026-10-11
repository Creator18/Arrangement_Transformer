#!/bin/bash
# Velocity labels for every rendered clip, from MIDI alone (no audio).  CPU.
#   sbatch ~/arrangement_transformer/backend/eval/velocity.sbatch
#SBATCH -J maestro-velocity
#SBATCH -p short
#SBATCH --array=0-31
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH -o /scratch/basu.ang/logs/velocity_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
export ARRANGEMENT_RENDER_DIR=/scratch/basu.ang/maestro_render_multi
cd ~/arrangement_transformer/backend
python -u eval/make_velocity_rolls.py --shard "${SLURM_ARRAY_TASK_ID}/32"

#!/bin/bash
# Render all of MAESTRO through the multi-piano pool on Explorer.
#
# CPU only: sfizz has no GPU path, so this belongs on `short`, not a GPU
# partition.  128 small tasks (2 cores, 4 GB each) rather than a few big ones,
# because small requests slot into partly-busy nodes and start sooner.
# Resumable: resubmitting re-renders only windows missing from disk.
#
#   sbatch ~/arrangement_transformer/backend/eval/render_multi.sbatch
#
#SBATCH -J maestro-render
#SBATCH -p short
#SBATCH --array=0-127
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH -o /scratch/basu.ang/logs/render_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
export ARRANGEMENT_RENDER_DIR=/scratch/basu.ang/maestro_render_multi
cd ~/arrangement_transformer/backend
python -u eval/render_multi_full.py --workers 2 --shard "${SLURM_ARRAY_TASK_ID}/128"

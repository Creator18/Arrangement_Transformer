#!/bin/bash
# Real MAESTRO recordings -> training clips, fetched straight from the archive
# by range requests.  CPU + network only.  Resumable.
#   sbatch --export=SPLIT=train,SHARDS=48 --array=0-47 fetch_real.sbatch
#   sbatch --export=SPLIT=validation,SHARDS=8 --array=0-7 fetch_real.sbatch
#SBATCH -J maestro-real
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH -o /scratch/basu.ang/logs/real_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
python -u eval/fetch_real_clips.py --split "$SPLIT" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}" \
    --out /scratch/basu.ang/maestro_real_clips

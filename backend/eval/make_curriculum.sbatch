#!/bin/bash
# Curriculum clips, one shard per array task.
#   sbatch --array=0-29 --export=ALL,KIND=vocal eval/make_curriculum.sbatch
#   sbatch --array=0-31 --export=ALL,KIND=slakh,STAGE=2 eval/make_curriculum.sbatch
#SBATCH -J curriculum
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/curriculum_%x_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
# numba's on-disk cache on the shared filesystem segfaults pYIN; keep it node-local
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
cd ~/arrangement_transformer/backend
OUT=${OUTROOT:-/scratch/basu.ang/curriculum}
N=${SLURM_ARRAY_TASK_COUNT}
if [ "$KIND" = vocal ]; then
  python -u eval/make_vocal_clips.py --musdb /scratch/basu.ang/datasets/musdb18hq --out $OUT \
      --windows 6 --shard ${SLURM_ARRAY_TASK_ID}/$N
else
  SLAKH=$(dirname "$(find /scratch/basu.ang/datasets/slakh2100_redux -maxdepth 3 -type d -name train | head -1)")
  python -u eval/make_curriculum_clips.py --slakh "$SLAKH" \
      --out $OUT --stage $STAGE --combos ${COMBOS:-3} --windows ${WINDOWS:-2} --shard ${SLURM_ARRAY_TASK_ID}/$N
fi

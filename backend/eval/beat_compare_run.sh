#!/bin/bash
module load miniconda3/25.9.1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
python eval/beat_compare.py --data /scratch/basu.ang/beat_data --out /scratch/basu.ang/beat_compare/madmom_$SLURM_ARRAY_TASK_ID.json \
    --shard $SLURM_ARRAY_TASK_ID --nshards 10 2>&1 | grep -v -i warn

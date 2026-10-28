#!/bin/bash
# one shard of eval/beat_labels.py: array tasks 0-79 = train shards, 80-89 = validation shards
module load miniconda3/25.9.1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 80 ]; then sp=train; sh=$i; n=80; else sp=validation; sh=$((i-80)); n=10; fi
python eval/beat_labels.py --pop909 /scratch/basu.ang/datasets/pop909_tmp/POP909 --clips /scratch/basu.ang/base_render \
    --split $sp --out /scratch/basu.ang/beat_data --shard $sh --nshards $n 2>&1 | grep -v -i "warn"

#!/bin/bash
# Lakh MIDI -> grammar corpus shards (piano-folded), one shard per array task.
#SBATCH -J lakh-corpus
#SBATCH -p short
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/lakh_corpus_%A_%a.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
python -u eval/make_grammar_corpus.py --source lakh --lakh /scratch/basu.ang/datasets/lmd/lmd_full \
    --out /scratch/basu.ang/grammar_corpus --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}

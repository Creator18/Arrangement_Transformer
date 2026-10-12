#!/bin/bash
# Grammar model pretraining (denoising: corrupted notes in, clean notes out).
#SBATCH -J grammar
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=03:15:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/grammar_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
python -u -m app.ml.grammar.train --corpus /scratch/basu.ang/grammar_corpus \
    --ckpt-dir ~/arrangement_transformer/checkpoints/grammar_v1 "$@"

#!/bin/bash
# Evaluate the trained spectral model: threshold sweep, test, per piano, baseline.
#   sbatch ~/arrangement_transformer/backend/eval/spectral_evaluate.sbatch
#SBATCH -J spectral-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/eval_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
python -u eval/spectral_evaluate.py \
    --ckpt ~/arrangement_transformer/checkpoints/spectral/best.pt \
    --data /scratch/basu.ang/maestro_render_multi \
    --procs 11 \
    --out ~/arrangement_transformer/checkpoints/spectral/eval.json

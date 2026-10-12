#!/bin/bash
# Soft-note fine-tune vs current real fine-tune on all 3,686 real test clips.
# ByteDance not re-run: 0.962 on these exact clips and scoring (job 10997342).
#SBATCH -J spectral-eval-soft
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/eval_soft_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
python -u eval/spectral_evaluate.py \
    --ckpt real-finetune=$C/spectral_real/best.pt \
    --ckpt soft=$C/spectral_soft/best.pt \
    --data /scratch/basu.ang/maestro_real_clips \
    --no-baseline \
    --procs 11 \
    --out $C/spectral_soft/eval_real.json

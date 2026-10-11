#!/bin/bash
# Full real-recording test: real fine-tune vs rendered-only vs ByteDance, on all
# 3,686 real MAESTRO test clips, unfiltered references, fixed decoder.
#   sbatch ~/arrangement_transformer/backend/eval/spectral_evaluate_real.sbatch
#SBATCH -J spectral-eval-real
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/eval_real_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
python -u eval/spectral_evaluate.py \
    --ckpt rendered-only=$C/spectral/best.pt \
    --ckpt real-finetune=$C/spectral_real/best.pt \
    --data /scratch/basu.ang/maestro_real_clips \
    --baseline-clips 0 \
    --procs 11 \
    --out $C/spectral_real/eval_real.json

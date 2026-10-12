#!/bin/bash
# Song-to-piano v2 vs v1 (and the untouched piano model) on the 106 held-out test songs.
#SBATCH -J s2p-eval-v2
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/s2p_eval_v2_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
python -u eval/s2p_evaluate.py --m2m /scratch/basu.ang/m2m \
    --ckpt s2p-v1=$C/s2p_v1/best.pt --ckpt s2p-v2=$C/s2p_v2/best.pt \
    --piano $C/spectral_real/best.pt --out $C/s2p_v2/eval.json

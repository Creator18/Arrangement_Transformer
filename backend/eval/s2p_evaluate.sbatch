#!/bin/bash
# Song-to-piano v1 evaluation on the 106 held-out Music2MIDI test songs.
#SBATCH -J s2p-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/s2p_eval_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
python -u eval/s2p_evaluate.py --m2m /scratch/basu.ang/m2m \
    --ckpt $C/s2p_v1/best.pt --piano $C/spectral_real/best.pt \
    --september $C/arranger.pt --out $C/s2p_v1/eval.json

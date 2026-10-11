#!/bin/bash
# Train the spectral patch transformer on one GPU.
#
# The gpu partition's limit is 8 h; train.py saves last.pt before then and
# resumes from it, so a long run is just this job resubmitted.  Extra
# arguments are passed straight to train.py (later flags override earlier):
#
#   sbatch --gres=gpu:a100:1 train.sbatch                 # prefer a fast GPU
#   sbatch --gres=gpu:v100-sxm2:1 train.sbatch            # fall back to V100
#   sbatch train.sbatch --total-steps 400 --ckpt-dir ~/arrangement_transformer/checkpoints/trial
#
#SBATCH -J spectral-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o /home/basu.ang/arrangement_transformer/logs/train_%j.out

module load miniconda3/25.9.1
source activate ~/envs/arr
cd ~/arrangement_transformer/backend
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u -m app.ml.spectral.train \
    --data /scratch/basu.ang/maestro_render_multi \
    --ckpt-dir ~/arrangement_transformer/checkpoints/spectral \
    --workers 7 "$@"

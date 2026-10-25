#!/bin/bash
# The singer adapter (stage C of eval/launch_singer.sh; its renders are in /scratch/.../arr_sing): LoRA on
# arr_tone_full over the singer batch + the level batch (+ half as much of the earlier mix), with the bass-
# masking weight (notes over the same note an octave or more below count 3x).  Then the real-song comparison.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_sing,$S/arr_level --old $S/ft_tone --out $S/ft_sing --old-frac 0.5
CPUS=12 bash eval/gpu_train.sh launch ft-sing $C/arr_sing_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_sing \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_sing_lora --mode lora --lora 16 --aux-weight 0.2 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 --octave-weight 3
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-sing -o $L/wait_sing_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_sing_lora sing arr_level_lora"

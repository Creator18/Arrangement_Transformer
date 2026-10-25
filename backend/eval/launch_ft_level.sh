#!/bin/bash
# After the level batch: build the fine-tune set (level renders + half as many from the earlier
# fine-tune mix, so tone / vocal / slide handling is kept), then train LoRA adapters on
# arr_tone_full through gpu_train.sh (any GPU now, moves to an A100 / H200 when one frees up).
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_level --old $S/ft_tone --out $S/ft_level --old-frac 0.5
CPUS=12 bash eval/gpu_train.sh launch ft-level $C/arr_level_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_level \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_level_lora --mode lora --lora 16 --aux-weight 0.2 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 2000

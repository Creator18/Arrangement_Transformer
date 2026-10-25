#!/bin/bash
# Quick-note adapter (tempo-up renders + 3x onset-loss weight on quick notes + oversampling): sped-up renders (POP909 at 1.2-2x tempo, exact labels) + the level mix, LoRA on
# arr_tone_full through gpu_train.sh (any GPU, moving to an A100 / H200 when one frees up), then the
# real-song comparison and the quick-note tests (normal-tempo and sped-up validation clips).
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_fast --old $S/ft_level --out $S/ft_fast --old-frac 1.0
CPUS=12 bash eval/gpu_train.sh launch ft-fast $C/arr_fast_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_fast \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_fast_lora --mode lora --lora 16 --aux-weight 0.2 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 --quick-weight 3 --oversample
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-fast -o $L/wait_fast_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_fast_lora fast arr_level_lora"

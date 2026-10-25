#!/bin/bash
# Attack adapter: the shared-note recipe (LoRA on arr_tone_full + note-end output; quick 3x / re-strike
# 1.5x / octave 3x; oversampling; stop on plateau) with the ATTACK-DELAY renders added: instruments'
# sound comes in up to 120 ms late and swells over up to 0.5 s, while the target keeps the written start.
# Data: attack batch + the shared-note sustain and tempo-up batches.  gpu_train.sh: any GPU, moving to an
# A100 / H200 when one frees up.  Then real songs, quick-note / double-press, and timing (by delay) tests.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_attack,$S/arr_sustain2,$S/arr_fast2 --old $S/arr_attack --out $S/ft_attack --old-frac 0.0
CPUS=12 bash eval/gpu_train.sh launch ft-attack $C/arr_attack_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_attack \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_attack_lora --mode lora --lora 16 --aux-weight 0.2 --offset-weight 1 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 \
  --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample
G="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-attack -o $L/wait_attack_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_attack_lora attack arr_union_lora; \
   for M in arr_attack_lora arr_union_lora; do \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J timing -o $L/timing_\${M}_attack_%j.out --wrap \"$G python -u eval/timing_check.py --ckpt $C/\$M/best.pt --clips $S/arr_attack/validation --n 600 --thr 0.92 --offset-thr 0.7\"; \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J ioi -o $L/ioi_\${M}_attack_%j.out --wrap \"$G python -u eval/ioi_check.py --ckpt $C/\$M/best.pt --clips $S/arr_attack/validation --n 800 --fine --no-variants\"; done"

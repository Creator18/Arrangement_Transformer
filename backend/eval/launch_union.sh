#!/bin/bash
# Shared-note adapter: same recipe as the sustain adapter (LoRA on arr_tone_full + NOTE-END output;
# quick 3x / re-strike 1.5x / octave 3x; oversampling; stop on plateau), but every POP909 render uses
# the shared-note convention (--union-shared): a note two parts share is ONE held note in the target.
# Data: the sustain batch + the tempo-up batch, both re-rendered that way (no older-convention data).
# gpu_train.sh: any GPU, moving to an A100 / H200 when one frees up.  Then: real songs, quick-note /
# double-press, timing, and the shared-note test, against the sustain adapter.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_sustain2,$S/arr_fast2 --old $S/arr_sustain2 --out $S/ft_union --old-frac 0.0
CPUS=12 bash eval/gpu_train.sh launch ft-union $C/arr_union_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_union \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_union_lora --mode lora --lora 16 --aux-weight 0.2 --offset-weight 1 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 \
  --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample
G="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-union -o $L/wait_union_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_union_lora union arr_sus_lora; \
   for M in arr_union_lora arr_sus_lora; do \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J timing -o $L/timing_\${M}_union_%j.out --wrap \"$G python -u eval/timing_check.py --ckpt $C/\$M/best.pt --clips $S/ft_union/validation --n 600 --offset-thr 0.7\"; \
     sbatch -p gpu --gres=gpu:1 -c 8 --mem=32G --time=02:00:00 -J shared -o $L/shared_\${M}_%j.out --wrap \"$G python -u eval/shared_notes.py --pop909 $S/datasets/pop909_tmp/POP909 --m2m $S/m2m --units $S/m2m/m2m_units.json --ckpt $C/\$M/best.pt --clips $S/ft_union/validation --n 300\"; \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J ioi -o $L/ioi_\${M}_union_%j.out --wrap \"$G python -u eval/ioi_check.py --ckpt $C/\$M/best.pt --clips $S/ft_union/validation --n 800 --fine --no-variants\"; done"

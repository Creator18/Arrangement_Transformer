#!/bin/bash
# Sustain / decay / release adapter: LoRA on arr_tone_full with a NOTE-END output, trained on the
# sustain-case renders (pedal, fades while held, long rings, dead stops, swells, dips, held vs
# re-struck) + the tempo-up renders + part of the level mix; quick-note 3x / re-strike 1.5x / octave
# 3x weighting and oversampling as before; stops on plateau.  gpu_train.sh: any GPU, moving to an
# A100 / H200 when one frees up.  Then: real-song comparison + quick-note and TIMING tests.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_sustain,$S/arr_fast --old $S/ft_level --out $S/ft_sus --old-frac 0.5
CPUS=12 bash eval/gpu_train.sh launch ft-sus $C/arr_sus_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_sus \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_sus_lora --mode lora --lora 16 --aux-weight 0.2 --offset-weight 1 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 \
  --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-sus -o $L/wait_sus_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_sus_lora sus arr_fast_lora; \
   for M in arr_sus_lora arr_fast_lora; do for D in ft_level arr_fast_test; do \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J timing -o $L/timing_\${M}_\${D}_%j.out --wrap \
       \"module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend; python -u eval/timing_check.py --ckpt $C/\$M/best.pt --clips $S/\$D/validation --n 600\"; done; done"

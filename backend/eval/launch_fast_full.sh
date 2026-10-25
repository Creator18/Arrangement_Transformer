#!/bin/bash
# Full fine-tune of arr_tone_full on the tempo-up + level mix: 3x onset weight on quick notes (different
# key), 1.5x on same-key re-strikes (fewer double presses), 3x on notes with the same note sounding 1-3
# octaves below (bass masking), oversampling, early stop on plateau.  gpu_train.sh: any GPU, moving to an
# A100 / H200 when one frees up.  When it finishes: real-song comparison + quick-note/octave tests, then
# the singer adapter (eval/launch_singer_train.sh).
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_fast --old $S/ft_level --out $S/ft_fast --old-frac 1.0      # adds the late renders
CPUS=12 bash eval/gpu_train.sh launch ft-fastfull $C/arr_fast_full $L python -u -m app.ml.spectral.ft_aux --data $S/ft_fast \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_fast_full --mode full --aux-weight 0.2 \
  --batch 16 --lr 3e-4 --total-steps 20000 --val-every 1000 --patience 3 \
  --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-fastfull -o $L/wait_fastfull_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_fast_full fastfull arr_level_lora; bash ~/arrangement_transformer/backend/eval/launch_singer_train.sh"

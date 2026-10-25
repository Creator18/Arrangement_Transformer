#!/bin/bash
# Heavy / vibrato / trill / roll adapter: the attack recipe (LoRA on arr_tone_full + note-end output; quick 3x /
# re-strike 1.5x / octave 3x; oversampling; stop on plateau) with the new batch: rock / EDM drums up to 12 dB over
# the band (kick on the bass notes, extra hits, genre sounds, sidechain), vibrato (one written note), trills and
# tremolo rolls (alternating notes), rolled / strummed chords (written chord).  Data: heavy batch + attack batch +
# shared-note tempo-up batch.  gpu_train.sh: any GPU, moving to an A100 / H200 when one frees up.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_heavy,$S/arr_attack,$S/arr_fast2 --old $S/arr_heavy --out $S/ft_heavy --old-frac 0.0
CPUS=12 bash eval/gpu_train.sh launch ft-heavy $C/arr_heavy_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_heavy \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_heavy_lora --mode lora --lora 16 --aux-weight 0.2 --offset-weight 1 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 \
  --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample
G="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-heavy -o $L/wait_heavy_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_heavy_lora heavy arr_attack_lora; \
   for M in arr_heavy_lora arr_attack_lora; do \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J ioi -o $L/ioi_\${M}_heavy_%j.out --wrap \"$G python -u eval/ioi_check.py --ckpt $C/\$M/best.pt --clips $S/arr_heavy/validation --n 800 --fine --no-variants\"; \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J timing -o $L/timing_\${M}_heavy_%j.out --wrap \"$G python -u eval/timing_check.py --ckpt $C/\$M/best.pt --clips $S/arr_heavy/validation --n 600 --thr 0.9 --offset-thr 0.7\"; done"

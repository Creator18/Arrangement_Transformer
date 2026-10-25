#!/bin/bash
# The combined adapter: every render case so far -- sustain / decay / release, attack delays and slow swells,
# levels, articulation both ways, sung melody, heavy rock / EDM drums + sidechain, vibrato, trills, tremolo and
# rolled chords -- with the user's shared-note rule (one strike if two voices start within 50 ms, else a
# re-strike; never release + re-press when the shorter note ends), + a tempo-up batch with the same rule.
# LoRA on arr_tone_full + note-end output; quick 3x / re-strike 1.5x / octave 3x; oversampling; stop on plateau.
# gpu_train.sh: any GPU, moving to an A100 / H200 when one frees up.  Then tests against the attack adapter.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
python -u eval/prep_ft_mix.py --new $S/arr_all,$S/arr_fast3 --old $S/arr_all --out $S/ft_all --old-frac 0.0
CPUS=12 bash eval/gpu_train.sh launch ft-all $C/arr_all_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_all \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_all_lora --mode lora --lora 16 --aux-weight 0.2 --offset-weight 1 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 \
  --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample
G="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-all -o $L/wait_all_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_all_lora all arr_attack_lora; \
   for M in arr_all_lora arr_attack_lora; do \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J ioi -o $L/ioi_\${M}_all_%j.out --wrap \"$G python -u eval/ioi_check.py --ckpt $C/\$M/best.pt --clips $S/arr_all/validation --n 800 --fine --no-variants\"; \
     sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=01:00:00 -J timing -o $L/timing_\${M}_all_%j.out --wrap \"$G python -u eval/timing_check.py --ckpt $C/\$M/best.pt --clips $S/arr_all/validation --n 600 --thr 0.9 --offset-thr 0.7\"; \
     sbatch -p gpu --gres=gpu:1 -c 8 --mem=32G --time=02:00:00 -J shared -o $L/shared_\${M}_all_%j.out --wrap \"$G python -u eval/shared_notes.py --pop909 $S/datasets/pop909_tmp/POP909 --m2m $S/m2m --units $S/m2m/m2m_units.json --ckpt $C/\$M/best.pt --clips $S/arr_all/validation --n 300\"; done"

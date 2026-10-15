#!/bin/bash
# After the tone batch: wait until the vocal fine-tune has left the queue (GPU submit limit),
# build the mixed fine-tune set, then launch both side-output fine-tunes through gpu_train.sh.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
while squeue -u $USER -h -n ft-vocal,ft-vocal-fast | grep -q .; do sleep 120; done
python -u eval/prep_ft_mix.py --new $S/arr_tone,$S/arr_vocal --old $S/arr_render --out $S/ft_tone --old-frac 0.33
for m in lora_stem full; do
  bash eval/gpu_train.sh launch ft-$m $C/arr_tone_$m $L python -u -m app.ml.spectral.ft_aux --data $S/ft_tone \
    --base $C/arr_scratch/best.pt --ckpt-dir $C/arr_tone_$m --mode $m --aux-weight 0.2 --total-steps 20000 --val-every 2000 --workers 7
done

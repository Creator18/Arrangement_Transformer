#!/bin/bash
# eval_when_done.sh <ckpt_dir> <label>: wait (on CPU) until the training there has finished, then
# submit the real-song comparison (new model vs the render-trained base vs v1) as a GPU job.
CK=$1; LABEL=$2; BASE=${3:-arr_scratch}; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs; S=/scratch/basu.ang
until tail -3 "$CK/train.log" 2>/dev/null | grep -q "(finished)"; do sleep 300; done
sbatch -p gpu --gres=gpu:1 -c 8 --mem=32G --time=03:00:00 -J eval-$LABEL -o $L/eval_${LABEL}_%j.out --wrap \
  "module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend; python -u eval/s2p_evaluate.py --m2m $S/m2m --units $S/m2m/m2m_units.json --exclude-old-train --ckpt $LABEL=$CK/best.pt --ckpt base=$C/$BASE/best.pt --ckpt v1=$C/s2p_v1/best.pt --piano $C/spectral_real/best.pt --out $S/eval_$LABEL.json"
# quick-note tests on held-out renders: normal tempo and sped up (eval/ioi_check.py)
for D in ft_level/validation arr_fast_test/validation; do
  sbatch -p gpu --gres=gpu:1 -c 4 --mem=24G --time=02:00:00 -J ioi-$LABEL -o $L/ioi_${LABEL}_$(basename $(dirname $D))_%j.out --wrap     "module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend; python -u eval/ioi_check.py --ckpt $CK/best.pt --clips $S/$D --n 800 --fine --no-variants"
done

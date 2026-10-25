#!/bin/bash
# One adapter per concept, each = the attack adapter's recipe + ONE change (user's ablation):
#   heavy  rock / EDM drums (old shared-note rule)       vib   vibrato (old rule)
#   orn    trills, tremolo + rolled chords (old rule)    rule  the user's revised shared-note rule only
# A, B, C train on their batch + the attack adapter's own data (ft_attack, old rule); D trains on new-rule
# data only (its batch + the new-rule tempo-up batch).  Tests vs the attack adapter: quick-note / double
# press, timing (by sound delay), shared-note re-presses, real songs.  GPU jobs are kept under the QOS limit.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
G="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
gpu() {   # gpu <name> <log> <command...>: submit when fewer than 6 of my GPU jobs are queued/running
  while [ $(squeue -u $USER -h -p gpu | wc -l) -ge 6 ]; do sleep 60; done
  sbatch --parsable -p gpu --gres=gpu:1 -c 12 --mem=48G --time=08:00:00 -J "$1" -o "$2" --wrap "$G ${*:3}"
}
waitj() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 120; done; }
declare -A TR
for n in heavy vib orn rule; do
  if [ $n = rule ]; then python -u eval/prep_ft_mix.py --new $S/abl_rule,$S/arr_fast3 --old $S/abl_rule --out $S/ft_abl_rule --old-frac 0.0
  else python -u eval/prep_ft_mix.py --new $S/abl_$n --old $S/ft_attack --out $S/ft_abl_$n --old-frac 2.0; fi
  TR[$n]=$(gpu abl-tr-$n $L/abl_train_${n}_%j.out python -u -m app.ml.spectral.ft_aux --data $S/ft_abl_$n \
    --base $C/arr_tone_full/best.pt --ckpt-dir $C/abl_$n --mode lora --lora 16 --aux-weight 0.2 --offset-weight 1 \
    --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 --workers 11 \
    --quick-weight 3 --restrike-weight 1.5 --octave-weight 3 --oversample)
  echo "train $n ${TR[$n]}"
done
for n in heavy vib orn rule; do
  waitj ${TR[$n]}; tail -2 $C/abl_$n/train.log
  CK=$C/abl_$n/best.pt
  gpu abl-ev-$n $L/abl_eval_${n}_%j.out python -u eval/s2p_evaluate.py --m2m $S/m2m --units $S/m2m/m2m_units.json --exclude-old-train \
    --ckpt $n=$CK --ckpt attack=$C/arr_attack_lora/best.pt --piano $C/spectral_real/best.pt --out $S/eval_abl_$n.json
  gpu abl-ioi-$n $L/abl_ioi_${n}_%j.out python -u eval/ioi_check.py --ckpt $CK --clips $S/abl_$n/validation --n 600 --fine --no-variants
  gpu abl-tim-$n $L/abl_timing_${n}_%j.out python -u eval/timing_check.py --ckpt $CK --clips $S/abl_$n/validation --n 600 --thr 0.9 --offset-thr 0.7
  gpu abl-sh-$n $L/abl_shared_${n}_%j.out python -u eval/shared_notes.py --pop909 $S/datasets/pop909_tmp/POP909 --m2m $S/m2m --units $S/m2m/m2m_units.json --ckpt $CK --clips $S/abl_$n/validation --n 300
  # the attack adapter on the same clips, for the side-by-side
  gpu abl-ioiA-$n $L/abl_ioi_attack_on_${n}_%j.out python -u eval/ioi_check.py --ckpt $C/arr_attack_lora/best.pt --clips $S/abl_$n/validation --n 600 --fine --no-variants
  gpu abl-timA-$n $L/abl_timing_attack_on_${n}_%j.out python -u eval/timing_check.py --ckpt $C/arr_attack_lora/best.pt --clips $S/abl_$n/validation --n 600 --thr 0.9 --offset-thr 0.7
  gpu abl-shA-$n $L/abl_shared_attack_on_${n}_%j.out python -u eval/shared_notes.py --pop909 $S/datasets/pop909_tmp/POP909 --m2m $S/m2m --units $S/m2m/m2m_units.json --ckpt $C/arr_attack_lora/best.pt --clips $S/abl_$n/validation --n 300
done
echo "(all submitted)"

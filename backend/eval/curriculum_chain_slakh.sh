#!/bin/bash
# Ablation: the curriculum WITHOUT the MUSDB vocal stage.  Confidence collapsed
# exactly at stage V in both curriculum runs; this tests whether removing it
# keeps the precision gain without the recall loss.  Same aligned Slakh clips
# (curriculum2), same settings, no MUSDB anywhere.  Checkpoints *_sk.
#   bash eval/curriculum_chain_slakh.sh
set -e
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
CU=/scratch/basu.ang/curriculum2
SP=app/ml/spectral
mix=$(sbatch --parsable -p short -c 1 --mem=2G --time=00:30:00 -J sk-mix \
  -o ~/arrangement_transformer/logs/sk_mix_%j.out --wrap "
for k in 2 3 6; do for s in train validation test; do
  d=$CU/slakhmix\$k/\$s; rm -rf \$d; mkdir -p \$d
  for st in 2 3 4 6; do [ \$st -le \$k ] && ln -sf $CU/stage\$st/\$s/*.np[yz] \$d/ 2>/dev/null || true; done
  echo slakhmix\$k \$s \$(ls \$d | grep -c npz)
done; done")
common="--data-real /scratch/basu.ang/maestro_real_clips --real-frac 0.2 --roles --lr 1e-4 --warmup 300 --val-every 2000 --val-clips 300"
s2=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J sk-S2 --dependency=afterok:$mix $SP/train.sbatch \
  --data $CU/slakhmix2 --init $C/spectral_real/best.pt --ckpt-dir $C/sk_S2 --total-steps 10000 $common)
s3=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J sk-S3 --dependency=afterok:$s2 $SP/train.sbatch \
  --data $CU/slakhmix3 --init $C/sk_S2/best.pt --ckpt-dir $C/sk_S3 --total-steps 10000 $common)
s6=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:15:00 -J sk-S46 --dependency=afterok:$s3 $SP/train.sbatch \
  --data $CU/slakhmix6 --init $C/sk_S3/best.pt --ckpt-dir $C/sk_S46 --total-steps 12000 $common)
arr=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:30:00 -J s2p-sk --dependency=afterok:$s6 $SP/train.sbatch \
  --data /scratch/basu.ang/m2m/clips --n-cond 4 --roles --init $C/sk_S46/best.pt --ckpt-dir $C/s2p_sk \
  --lr 1e-4 --warmup 500 --total-steps 25000 --val-every 2000 --val-clips 300)
ev=$(sbatch --parsable -J s2p-eval-sk --dependency=afterok:$arr eval/s2p_evaluate_sk.sbatch)
g=$(sbatch --parsable -p gpu --gres=gpu:v100-sxm2:1 -c 4 --mem=24G --time=00:20:00 -J confsk-gpu --dependency=afterok:$arr \
  -o ~/arrangement_transformer/logs/confsk_gpu_%j.out eval/confidence_trace_sk.sh gpu)
c=$(sbatch --parsable -p short -c 2 --mem=16G --time=01:00:00 -J confsk-cpu --dependency=afterok:$g \
  -o ~/arrangement_transformer/logs/confsk_cpu_%j.out eval/confidence_trace_sk.sh cpu)
echo "mix $mix S2 $s2 S3 $s3 S46 $s6 arranger $arr eval $ev conf $g $c"

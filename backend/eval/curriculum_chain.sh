#!/bin/bash
# Song-to-piano v2 curriculum, run end to end with Slurm dependencies.
# Run on the login node AFTER Slakh is extracted:  bash eval/curriculum_chain.sh
#   1. Slakh clips for stages 2, 3, 4, 6                       (CPU arrays, parallel)
#   2. stage mixes: stage k = Slakh stages <= k + all MUSDB vocal clips (symlinks)
#   3. cur_S2 <- cur_V, cur_S3 <- cur_S2, cur_S46 <- cur_S3     (GPU, chained)
#   4. s2p_v2: Music2MIDI arranger fine-tune <- cur_S46, difficulty + roles
#   5. s2p_v2 evaluation on the 106 held-out songs, beside v1
set -e
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
CU=/scratch/basu.ang/curriculum
SP=app/ml/spectral
clips=""
for k in 2 3 4 6; do
  j=$(sbatch --parsable -J cur-s$k --array=0-31 --export=ALL,KIND=slakh,STAGE=$k eval/make_curriculum.sbatch)
  clips="$clips:$j"
done
mix=$(sbatch --parsable -p short -c 1 --mem=2G --time=00:20:00 -J cur-mix --dependency=afterok$clips \
  -o ~/arrangement_transformer/logs/cur_mix_%j.out --wrap "
for k in 2 3 6; do for s in train validation test; do
  d=$CU/mix\$k/\$s; rm -rf \$d; mkdir -p \$d
  for st in 2 3 4 6; do [ \$st -le \$k ] && ln -sf $CU/stage\$st/\$s/*.np[yz] \$d/ 2>/dev/null || true; done
  ln -sf $CU/vocal/\$s/*.npz \$d/
  echo mix\$k \$s \$(ls \$d | grep -c npz)
done; done")
common="--data-real /scratch/basu.ang/maestro_real_clips --real-frac 0.2 --roles --lr 1e-4 --warmup 300 --val-every 2000 --val-clips 300"
s2=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J cur-S2 --dependency=afterok:$mix $SP/train.sbatch \
  --data $CU/mix2 --init $C/cur_V/best.pt --ckpt-dir $C/cur_S2 --total-steps 10000 $common)
s3=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J cur-S3 --dependency=afterok:$s2 $SP/train.sbatch \
  --data $CU/mix3 --init $C/cur_S2/best.pt --ckpt-dir $C/cur_S3 --total-steps 10000 $common)
s6=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:15:00 -J cur-S46 --dependency=afterok:$s3 $SP/train.sbatch \
  --data $CU/mix6 --init $C/cur_S3/best.pt --ckpt-dir $C/cur_S46 --total-steps 12000 $common)
arr=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:30:00 -J s2p-v2 --dependency=afterok:$s6 $SP/train.sbatch \
  --data /scratch/basu.ang/m2m/clips --n-cond 4 --roles --init $C/cur_S46/best.pt --ckpt-dir $C/s2p_v2 \
  --data-real $CU/vocal --real-frac 0.1 --lr 1e-4 --warmup 500 --total-steps 25000 --val-every 2000 --val-clips 300)
ev=$(sbatch --parsable -J s2p-eval-v2 --dependency=afterok:$arr --export=ALL eval/s2p_evaluate_v2.sbatch)
echo "clips$clips mix $mix S2 $s2 S3 $s3 S46 $s6 arranger $arr eval $ev"

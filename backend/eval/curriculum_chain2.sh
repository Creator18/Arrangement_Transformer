#!/bin/bash
# Song-to-piano v3: the curriculum again, with ALIGNED labels (see
# eval/label_alignment_check.py): MUSDB pYIN onsets snapped to the isolated
# stems' attacks (unsure ones masked), Slakh shifted per instrument class.
# Writes to /scratch/basu.ang/curriculum2, checkpoints *_v3 — v2's are kept.
#   bash eval/curriculum_chain2.sh
set -e
cd ~/arrangement_transformer/backend
C=~/arrangement_transformer/checkpoints
CU=/scratch/basu.ang/curriculum2
SP=app/ml/spectral
mkdir -p $CU/vocal && cp -n /scratch/basu.ang/curriculum/vocal/labels_*.npz $CU/vocal/ 2>/dev/null || true
voc=$(sbatch --parsable -J cur2-vocal --array=0-29 --export=ALL,KIND=vocal,OUTROOT=$CU eval/make_curriculum.sbatch)
clips=":$voc"
for k in 2 3 4 6; do
  j=$(sbatch --parsable -J cur2-s$k --array=0-31 --export=ALL,KIND=slakh,STAGE=$k,OUTROOT=$CU eval/make_curriculum.sbatch)
  clips="$clips:$j"
done
chk=$(sbatch --parsable -p short -c 2 --mem=8G --time=01:00:00 -J align-check2 --dependency=afterok$clips \
  -o ~/arrangement_transformer/logs/align_check2_%j.out --wrap "module load miniconda3/25.9.1; source activate ~/envs/arr; \
  export NUMBA_CACHE_DIR=/tmp/numba_\$SLURM_JOB_ID; python -u eval/label_alignment_check.py \
  --slakh \$(dirname \$(find /scratch/basu.ang/datasets/slakh2100_redux -maxdepth 3 -type d -name train | head -1)) \
  --slakh-shift --musdb /scratch/basu.ang/datasets/musdb18hq --labels $CU/vocal --labels-prefix labels2_ \
  --maestro-clips /scratch/basu.ang/maestro_real_clips")
mix=$(sbatch --parsable -p short -c 1 --mem=2G --time=00:30:00 -J cur2-mix --dependency=afterok$clips \
  -o ~/arrangement_transformer/logs/cur2_mix_%j.out --wrap "
for k in 2 3 6; do for s in train validation test; do
  d=$CU/mix\$k/\$s; rm -rf \$d; mkdir -p \$d
  for st in 2 3 4 6; do [ \$st -le \$k ] && ln -sf $CU/stage\$st/\$s/*.np[yz] \$d/ 2>/dev/null || true; done
  ln -sf $CU/vocal/\$s/*.npz \$d/
  echo mix\$k \$s \$(ls \$d | grep -c npz)
done; done")
common="--data-real /scratch/basu.ang/maestro_real_clips --real-frac 0.2 --roles --lr 1e-4 --warmup 300 --val-every 2000 --val-clips 300"
v=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J cur3-V --dependency=afterok:$mix $SP/train.sbatch \
  --data $CU/vocal --data-real /scratch/basu.ang/maestro_real_clips --real-frac 0.3 --roles \
  --init $C/spectral_real/best.pt --ckpt-dir $C/cur3_V --lr 1e-4 --warmup 500 --total-steps 12000 --val-every 2000 --val-clips 300)
s2=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J cur3-S2 --dependency=afterok:$v $SP/train.sbatch \
  --data $CU/mix2 --init $C/cur3_V/best.pt --ckpt-dir $C/cur3_S2 --total-steps 10000 $common)
s3=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:00:00 -J cur3-S3 --dependency=afterok:$s2 $SP/train.sbatch \
  --data $CU/mix3 --init $C/cur3_S2/best.pt --ckpt-dir $C/cur3_S3 --total-steps 10000 $common)
s6=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:15:00 -J cur3-S46 --dependency=afterok:$s3 $SP/train.sbatch \
  --data $CU/mix6 --init $C/cur3_S3/best.pt --ckpt-dir $C/cur3_S46 --total-steps 12000 $common)
arr=$(sbatch --parsable --gres=gpu:v100-sxm2:1 --time=01:30:00 -J s2p-v3 --dependency=afterok:$s6 $SP/train.sbatch \
  --data /scratch/basu.ang/m2m/clips --n-cond 4 --roles --init $C/cur3_S46/best.pt --ckpt-dir $C/s2p_v3 \
  --data-real $CU/vocal --real-frac 0.1 --lr 1e-4 --warmup 500 --total-steps 25000 --val-every 2000 --val-clips 300)
ev=$(sbatch --parsable -J s2p-eval-v3 --dependency=afterok:$arr eval/s2p_evaluate_v3.sbatch)
echo "vocal $voc clips$clips check $chk mix $mix V $v S2 $s2 S3 $s3 S46 $s6 arranger $arr eval $ev"

#!/bin/bash
# The modular pipeline (user's design): piano transcriber -> PLAIN adapter -> fold -> one thin adapter per effect.
# Everything is submitted up front with dependencies (no waiter jobs).  Renders spread over two CPU partitions
# (short + the course partition), 1 core each.  Training runs in 6 GPU "lanes" (4 on gpu, 2 on courses-gpu), each
# training its modules back to back and testing each against plain-base right after: GPUs never wait on small jobs.
cd ~/arrangement_transformer/backend
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs
ENV="module load miniconda3/25.9.1; source activate ~/envs/arr; export NUMBA_CACHE_DIR=/tmp/nb_\$SLURM_JOB_ID_\$SLURM_ARRAY_TASK_ID; cd ~/arrangement_transformer/backend;"
R="python -u eval/render_modular.py --pop909 $S/datasets/pop909_tmp/POP909 --sf2 $S/datasets/OmegaGMGS2.sf2"
render() {  # render <module> <renders/song> <partition>  -> array job id (100 shards, 1 core each)
  local acct=""; [ "$3" = courses ] && acct="-A cs7170.202710"
  sbatch --parsable $acct -p $3 -c 1 --mem=6G --time=04:00:00 --array=0-99%${4:-25} -J mr-$1 -o $L/mr_$1_%A_%a.out \
    --wrap "$ENV $R --module $1 --renders $2 --out $S/mod_$1 --tag x --shard \$SLURM_ARRAY_TASK_ID/100"
}
# ---- plain first, on both partitions' worth of slots
P=$(render plain 3 courses 60)
# ---- modules: wave 1 first (the user's biggest problems), alternating partitions
WAVE=(late glides connected noteend articulation decay expression vibrato rolls unison levels exposed singer timbre reverb eq dynamics drums heavydrums sidechain genre fast)
declare -A RJ; k=0
for m in "${WAVE[@]}"; do
  part=$([ $((k % 2)) -eq 0 ] && echo short || echo courses); RJ[$m]=$(render $m 2 $part 25); k=$((k + 1))
done
# ---- plain adapter on the piano transcriber, then fold it in -> plain_base.pt
TRAIN="python -u -m app.ml.spectral.ft_aux --mode lora --lora 16 --aux-weight 0 --offset-weight 0 --batch 16 --lr 4e-4 --total-steps 20000 --val-every 1000 --patience 3 --workers 11 --max-minutes 400"
PL=$(sbatch --parsable --dependency=afterany:$P -p gpu --gres=gpu:1 -c 12 --mem=48G --time=08:00:00 -J mod-plain -o $L/mod_plain_%j.out --wrap \
  "$ENV python -u eval/prep_ft_mix.py --new $S/mod_plain --old $S/mod_plain --out $S/ft_mod_plain --old-frac 0.0 && \
   $TRAIN --data $S/ft_mod_plain --base $C/spectral_real/best.pt --ckpt-dir $C/mod_plain && \
   python -u eval/merge_lora.py $C/mod_plain/best.pt $C/plain_base.pt && \
   python -u eval/ioi_check.py --ckpt $C/plain_base.pt --clips $S/mod_plain/validation --n 400 --fine --no-variants && \
   python -u eval/timing_check.py --ckpt $C/plain_base.pt --clips $S/mod_plain/validation --n 400 --thr 0.9")
echo "plain renders $P, plain adapter $PL"
# ---- 6 lanes: each trains its modules on plain-base and tests each (module adapter vs plain-base, same clips)
LANES=("late connected articulation expression" "glides noteend decay vibrato" "rolls unison levels exposed" "singer timbre reverb eq" "dynamics drums heavydrums" "sidechain genre fast")
for li in 0 1 2 3 4 5; do
  deps="afterok:$PL"; body=""
  for m in ${LANES[$li]}; do
    deps="$deps:${RJ[$m]}"
    body="$body python -u eval/prep_ft_mix.py --new $S/mod_$m --old $S/mod_plain --out $S/ft_mod_$m --old-frac 0.5; \
      $TRAIN --data $S/ft_mod_$m --base $C/plain_base.pt --ckpt-dir $C/mod_$m; \
      for CK in $C/mod_$m/best.pt $C/plain_base.pt; do \
        python -u eval/ioi_check.py --ckpt \$CK --clips $S/mod_$m/validation --n 400 --fine --no-variants > $L/modtest_ioi_${m}_\$(basename \$(dirname \$CK))_\$(basename \$CK .pt).out 2>&1; \
        python -u eval/timing_check.py --ckpt \$CK --clips $S/mod_$m/validation --n 400 --thr 0.9 > $L/modtest_timing_${m}_\$(basename \$(dirname \$CK))_\$(basename \$CK .pt).out 2>&1; done;"
  done
  if [ $li -lt 4 ]; then where="-p gpu"; else where="-A cs7170.202710 -p courses-gpu"; fi
  echo "lane $li (${LANES[$li]}): $(sbatch --parsable --dependency=$deps $where --gres=gpu:1 -c 12 --mem=48G --time=1-00:00:00 -J mod-lane$li -o $L/mod_lane${li}_%j.out --wrap "$ENV $body")"
done

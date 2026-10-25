#!/bin/bash
# Move the running base training onto a FASTER GPU (A100 / H200) the moment one is free (the user's rule: V100
# minimum, take better when available).  Submitted twice, once per GPU type; whichever starts first:
#   1. cancels the other request; makes the queued V100 spares (base-train-*) wait until THIS job ends, so two
#      trainers never share a folder -- and if this job dies (an A100 job was killed mid-run, 2026-10-25), the
#      spares resume from last.pt instead of the run going idle
#   2. asks the running trainer to save and stop (HANDOVER file; app/ml/spectral/train.py), waits for it to exit
#   3. resumes from last.pt with the identical command
# The tests stay chained after the last spare (a spare that finds the run finished exits at once).
# If the run already finished, it exits at once.
#
#   sbatch ... --gres=gpu:a100:1 -J base-fast eval/base_fast.sh <ckpt dir> <test job id> <train command...>
CK=$1; TEST=$2; shift 2
CMD=("$@")
echo "[fast] $(nvidia-smi --query-gpu=name --format=csv,noheader) on $(hostname)"
if tail -3 "$CK/train.log" 2>/dev/null | grep -q "(finished)"; then echo "[fast] training already finished"; exit 0; fi
for j in $(squeue -u "$USER" -h -n base-fast -o %i | grep -v "^${SLURM_JOB_ID}$"); do scancel "$j"; done
for j in $(squeue -u "$USER" -h -t PD -o "%i %j" | awk '$2 ~ /^base-train-/ {print $1}' | sort -n | head -1); do
  scontrol update jobid="$j" Dependency="afterany:$SLURM_JOB_ID" && echo "[fast] spare $j now waits for job $SLURM_JOB_ID"
done
RUN=$(squeue -u "$USER" -h -t R -o "%i %j" | awk '$2 ~ /^base-train-/ {print $1}' | head -1)
if [ -n "$RUN" ]; then
  touch "$CK/HANDOVER"; echo "[fast] asking job $RUN to save and stop"
  while squeue -j "$RUN" -h 2>/dev/null | grep -q .; do sleep 15; done
fi
rm -f "$CK/HANDOVER"
if tail -3 "$CK/train.log" 2>/dev/null | grep -q "(finished)"; then echo "[fast] training already finished"; exit 0; fi
module load miniconda3/25.9.1; source activate ~/envs/arr; export NUMBA_CACHE_DIR=/tmp/nb_$SLURM_JOB_ID
cd ~/arrangement_transformer/backend
"${CMD[@]}"

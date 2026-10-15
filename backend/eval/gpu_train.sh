#!/bin/bash
# Opportunistic GPU training.  Start on whatever GPU is free; keep requests open for a faster
# one (A100, H200).  When a faster GPU starts, it asks the running trainer to save and stop
# (HANDOVER file) and continues from last.pt.  A job that hits the 8 h limit unfinished
# resubmits the set.  At most MAX_SUBMITS submissions in total, so a fault can never loop.
#
#   gpu_train.sh launch  <name> <ckpt_dir> <log_dir> <train command...>    submit the set
#   gpu_train.sh run     <name> <ckpt_dir> <log_dir> <train command...>    (inside a job)
#   gpu_train.sh fast    <name> <ckpt_dir> <log_dir> <train command...>    (inside a faster-GPU job)
MODE=$1; NAME=$2; CK=$3; LOG=$4; shift 4
CMD=("$@")                                   # the training command, kept apart from everything else
SELF=$(readlink -f "$0")
MAX_SUBMITS=${MAX_SUBMITS:-24}
ENV="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
mkdir -p "$CK" "$LOG"

submit() {   # submit <gres> <mode> <job name>
  local n; n=$(cat "$CK/.submits" 2>/dev/null || echo 0)
  if [ "$n" -ge "$MAX_SUBMITS" ]; then echo "[gpu] submission limit ($MAX_SUBMITS) reached — not resubmitting"; return 1; fi
  echo $((n + 1)) > "$CK/.submits"
  sbatch --parsable -p gpu --gres="$1" -c 8 --mem=48G --time=08:00:00 -J "$3" -o "$LOG/$3_%j.out" \
    --wrap "$ENV bash $SELF $2 $NAME $CK $LOG $(printf '%q ' "${CMD[@]}")"
}
finished() { tail -3 "$CK/train.log" 2>/dev/null | grep -q "(finished)"; }
others() { squeue -u "$USER" -h -n "$1" -o %i | grep -v "^${SLURM_JOB_ID:-none}$"; }
relaunch() {
  finished && { echo "[gpu] training finished"; return; }
  [ -n "$(others "$NAME")" ] || submit gpu:1 run "$NAME"
  [ -n "$(others "${NAME}-fast")" ] || { submit gpu:a100:1 fast "${NAME}-fast"; submit gpu:h200:1 fast "${NAME}-fast"; }
}

case $MODE in
  launch)
    rm -f "$CK/.submits" "$CK/HANDOVER"; relaunch
    squeue -u "$USER" -h -n "$NAME,${NAME}-fast" -o "%i %j %T %b" ;;
  run)
    nvidia-smi --query-gpu=name --format=csv,noheader
    "${CMD[@]}"; rc=$?
    if [ -f "$CK/HANDOVER" ]; then echo "[gpu] handed over to a faster GPU"
    elif [ $rc -ne 0 ]; then echo "[gpu] trainer exited with code $rc — NOT resubmitting (fix the fault first)"
    else relaunch; fi ;;
  fast)
    echo "[gpu] faster GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    for j in $(others "${NAME}-fast"); do scancel "$j"; done
    for j in $(squeue -u "$USER" -h -n "$NAME" -t PD -o %i); do scancel "$j"; done
    RUN=$(squeue -u "$USER" -h -n "$NAME" -t R -o %i | head -1)
    if [ -n "$RUN" ]; then
      touch "$CK/HANDOVER"; echo "[gpu] asking job $RUN to save and stop"
      while squeue -j "$RUN" -h 2>/dev/null | grep -q .; do sleep 20; done
    fi
    rm -f "$CK/HANDOVER"
    finished && { echo "[gpu] training already finished"; exit 0; }
    "${CMD[@]}"; rc=$?
    if [ $rc -ne 0 ]; then echo "[gpu] trainer exited with code $rc — NOT resubmitting"
    elif ! finished; then submit gpu:a100:1 fast "${NAME}-fast"; submit gpu:h200:1 fast "${NAME}-fast"; submit gpu:1 run "$NAME"; fi ;;
esac

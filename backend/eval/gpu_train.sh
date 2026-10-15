#!/bin/bash
# Opportunistic GPU training.  Start on whatever GPU is free; keep requests open for a faster
# one (A100, H200).  When a faster GPU starts, it asks the running trainer to save and stop,
# and continues from last.pt.  When a job hits the 8 h limit unfinished, the trio is resubmitted.
#
#   gpu_train.sh launch  <name> <ckpt_dir> <log_dir> <train command...>    submit the trio
#   gpu_train.sh run     <name> <ckpt_dir> <log_dir> <train command...>    (inside a job)
#   gpu_train.sh fast    <name> <ckpt_dir> <log_dir> <train command...>    (inside a faster-GPU job)
MODE=$1; NAME=$2; CK=$3; LOG=$4; shift 4
SELF=$(readlink -f "$0")
ENV="module load miniconda3/25.9.1; source activate ~/envs/arr; cd ~/arrangement_transformer/backend;"
q() { printf '%q ' "$@"; }
submit() {   # $1 = gres, $2 = mode, $3 = job name
  sbatch --parsable -p gpu --gres=$1 -c 8 --mem=48G --time=08:00:00 -J "$3" -o "$LOG/$3_%j.out" \
    --wrap "$ENV bash $SELF $2 $NAME $CK $LOG $(q "$@")"
}
finished() { tail -3 "$CK/train.log" 2>/dev/null | grep -q "finished"; }
relaunch() {
  finished && { echo "[gpu] training finished"; return; }
  others() { squeue -u $USER -h -n "$1" -o %i | grep -v "^${SLURM_JOB_ID:-none}$"; }
  [ -n "$(others "$NAME")" ] || submit gpu:1 run "$NAME" "$@" >/dev/null
  [ -n "$(others "${NAME}-fast")" ] || { submit gpu:a100:1 fast "${NAME}-fast" "$@" >/dev/null; submit gpu:h200:1 fast "${NAME}-fast" "$@" >/dev/null; }
  echo "[gpu] resubmitted"
}
case $MODE in
  launch) mkdir -p "$CK" "$LOG"; relaunch "$@" ; squeue -u $USER -h -n "$NAME,${NAME}-fast" -o "%i %j %T %b" ;;
  run)    nvidia-smi --query-gpu=name --format=csv,noheader; "$@"; [ -f "$CK/HANDOVER" ] || relaunch "$@" ;;
  fast)   echo "[gpu] faster GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
          for j in $(squeue -u $USER -h -n "${NAME}-fast" -o %i); do [ "$j" != "$SLURM_JOB_ID" ] && scancel "$j"; done
          for j in $(squeue -u $USER -h -n "$NAME" -t PD -o %i); do scancel "$j"; done
          RUN=$(squeue -u $USER -h -n "$NAME" -t R -o %i | head -1)
          if [ -n "$RUN" ]; then touch "$CK/HANDOVER"; while squeue -j "$RUN" -h 2>/dev/null | grep -q .; do sleep 20; done; fi
          rm -f "$CK/HANDOVER"; "$@"
          finished || { submit gpu:a100:1 fast "${NAME}-fast" "$@" >/dev/null; submit gpu:h200:1 fast "${NAME}-fast" "$@" >/dev/null; submit gpu:1 run "$NAME" "$@" >/dev/null; } ;;
esac

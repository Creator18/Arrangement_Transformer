#!/bin/bash
# Stops a random-curriculum run at the first VAL check with identify F1 >= BAR, or at the first check at step >= MAX_STEP,
# whichever comes first: keeps that check's last.pt as stop_step<N>.pt, cancels the trainer, sets the producers' stop flag.
#   sbatch -p courses -A cs7170.202710 --export=ALL,JOB=<trainer job id>,MAX_STEP=30000,BAR=0.98 random_stop_watch.sh
#SBATCH -J rnd-stopwatch -c 1 --mem=1G -t 8:00:00 -o /home/basu.ang/arrangement_transformer/logs/rnd_stopwatch.out
RUN=${RUN:-/home/basu.ang/arrangement_transformer/checkpoints/arr_random2}
CTL=${CTL:-/scratch/basu.ang/random_curriculum/control.json}
MAX_STEP=${MAX_STEP:-30000}; BAR=${BAR:-0.98}
module load miniconda3/25.9.1; source activate ~/envs/arr   # torch, to read the step inside a checkpoint
seen=$(grep -c "^VAL step" $RUN/train.log)
echo "$(date) watching $RUN (job $JOB): stop at step >= $MAX_STEP or F1 >= $BAR; $seen checks already logged"
while squeue -h -j $JOB | grep -q .; do
  n=$(grep -c "^VAL step" $RUN/train.log)
  if [ "$n" -gt "$seen" ]; then
    line=$(grep "^VAL step" $RUN/train.log | tail -1); seen=$n
    step=$(echo "$line" | sed -E 's/^VAL step ([0-9]+).*/\1/')
    f1=$(echo "$line" | sed -E 's/.*"f1": ([0-9.]+).*/\1/')
    echo "$(date) check at step $step: F1 $f1"
    if [ "$step" -ge "$MAX_STEP" ] || awk "BEGIN{exit !($f1 >= $BAR)}"; then
      # the trainer logs the check, then saves last.pt (atomic replace): copy it until the copy holds this step
      # (by the step stored inside, not file times: the trainer's node and this one disagree on the clock)
      while :; do
        cp $RUN/last.pt $RUN/stop_step${step}.pt
        got=$(python -c "import torch; print(torch.load('$RUN/stop_step${step}.pt', map_location='cpu', weights_only=False)['step'])")
        [ "$got" -ge "$step" ] && break; sleep 5
      done
      scancel $JOB
      [ "$got" -eq "$step" ] || echo "$(date) WARNING: the copy holds step $got, not $step"
      level=$(sed -E 's/.*"level": *([0-9]+).*/\1/' $CTL)
      echo "{\"level\": $level, \"stop\": true}" > $CTL.tmp && mv $CTL.tmp $CTL
      echo "$(date) stopped at step $step (F1 $f1): kept stop_step${step}.pt, cancelled $JOB, producers told to stop"
      exit 0
    fi
  fi
  sleep 30
done
echo "$(date) trainer $JOB ended on its own"

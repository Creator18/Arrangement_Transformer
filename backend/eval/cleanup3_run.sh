#!/bin/bash
# One piece of the clean-up v3 fine-tune (app/ml/spectral/cleanup3_train.py).  The user's rule: start on any V100+,
# move to an H200 / A100 the moment one frees up, on gpu or gpu-short.
#   cleanup3_run.sh slow   any allowed GPU; exits at once if another piece holds the lock and is running
#   cleanup3_run.sh fast   an H200 / A100 waiter: cancels the other waiters, asks the running piece to save and exit
#                          (HANDOVER), waits for it, then resumes from last.pt
# A piece that stops on its time budget resubmits the next slow piece and fresh fast waiters (cleanup3_submit.sh).
KIND=$1
OUT=~/arrangement_transformer/checkpoints/cleanup3
LOCK=$OUT/lock
mkdir -p $OUT
module load miniconda3/25.9.1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
echo "$(date) $KIND piece $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"

holder=$(cat $LOCK 2>/dev/null)
if [ -n "$holder" ] && [ "$holder" != "$SLURM_JOB_ID" ] && squeue -h -j "$holder" -t R 2>/dev/null | grep -q .; then
    if [ "$KIND" = slow ]; then
        echo "piece $holder is training: nothing to do"; exit 0
    fi
    scancel -n cutrain3-fast -t PENDING
    echo "asking $holder to hand over"
    touch $OUT/HANDOVER
    while squeue -h -j "$holder" 2>/dev/null | grep -q .; do sleep 15; done
fi
[ "$KIND" = fast ] && scancel -n cutrain3-fast -t PENDING
echo $SLURM_JOB_ID > $LOCK
rm -f $OUT/HANDOVER

python -m app.ml.spectral.cleanup3_train --data /scratch/basu.ang/cleanup_data,/scratch/basu.ang/cleanup_data_tr \
    --beats /scratch/basu.ang/clip_beats --labels /scratch/basu.ang/merge_labels \
    --init ~/arrangement_transformer/checkpoints/cleanup2/best.pt \
    --split ~/arrangement_transformer/checkpoints/cleanup1/split.json --out $OUT --budget-min ${BUDGET:-108} \
    2>&1 | grep --line-buffered -v -i "warn\|pkg_resources" | tee $OUT/piece_$SLURM_JOB_ID.log

if grep -q "^done\|already finished" $OUT/piece_$SLURM_JOB_ID.log; then
    echo "finished: cancelling the waiters"
    scancel -n cutrain3-fast; scancel -n cutrain3-slow -t PENDING
elif grep -q "budget reached" $OUT/piece_$SLURM_JOB_ID.log; then
    echo "budget reached: submitting the next piece"
    bash ~/arrangement_transformer/backend/eval/cleanup3_submit.sh
fi

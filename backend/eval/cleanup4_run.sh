#!/bin/bash
# One piece of the clean-up v4 fine-tune (app/ml/spectral/cleanup4_train.py): < 2 h on gpu-short, then it submits the
# next piece itself until the run is done.
# env: OUT (checkpoint dir), INIT (start from), EXTRA (more trainer flags), JOB (name for the next piece)
OUT=${OUT:-~/arrangement_transformer/checkpoints/cleanup4}
INIT=${INIT:-~/arrangement_transformer/checkpoints/cleanup3/best.pt}
JOB=${JOB:-cutrain4}
mkdir -p $OUT
module load miniconda3/25.9.1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
echo "$(date) piece $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m app.ml.spectral.cleanup4_train --data /scratch/basu.ang/cleanup_data,/scratch/basu.ang/cleanup_data_tr \
    --beats /scratch/basu.ang/clip_beats --labels /scratch/basu.ang/merge_labels --human /scratch/basu.ang/human_data \
    --songs /scratch/basu.ang/never_kept/songs.json --init $INIT \
    --split ~/arrangement_transformer/checkpoints/cleanup1/split.json --out $OUT --budget-min ${BUDGET:-100} $EXTRA \
    2>&1 | grep --line-buffered -v -i "warn\|pkg_resources" | tee $OUT/piece_$SLURM_JOB_ID.log
if grep -q "budget reached" $OUT/piece_$SLURM_JOB_ID.log; then
    X=c2184,c2185,c2186,c2187,c2188,c2193,c2194,c2195,d1025,d1028,c2205
    bash eval/sbatch_once.sh -J $JOB -p gpu-short --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 -x $X \
        -o ~/arrangement_transformer/logs/${JOB}_%j.out eval/cleanup4_run.sh
fi

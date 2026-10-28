#!/bin/bash
# beat model training (app/ml/spectral/beat_train.py) in pieces of < 2 h (any V100+ on gpu or gpu-short); a piece that
# stops on its time budget resubmits the next one, which resumes from last.pt.
module load miniconda3/25.9.1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
OUT=~/arrangement_transformer/checkpoints/beat1
mkdir -p $OUT
echo "$(date) piece $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m app.ml.spectral.beat_train --data /scratch/basu.ang/beat_data --out $OUT --steps 40000 --budget-min 105 \
    2>&1 | grep --line-buffered -v -i "warn" | tee -a $OUT/train.log
if grep -q "budget reached" <(tail -3 $OUT/train.log); then
    bash eval/sbatch_once.sh -J beattrain -p gpu,gpu-short --gres=gpu:1 -c 8 --mem=48G -t 2:00:00 \
        -x c2184,c2185,c2186,c2187,c2188,c2193,c2194,c2195,d1025,d1028,c2205 \
        -o ~/arrangement_transformer/logs/beattrain_%j.out eval/beat_run.sh
fi

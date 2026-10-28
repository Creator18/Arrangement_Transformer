#!/bin/bash
# Submit one slow piece (first V100+ on gpu or gpu-short) and four fast waiters (H200 / A100 on gpu and gpu-short)
# for the clean-up v3 fine-tune; eval/cleanup3_run.sh does the hand-over.
cd ~/arrangement_transformer/backend
R=~/arrangement_transformer/backend/eval/cleanup3_run.sh
L=~/arrangement_transformer/logs
X=c2184,c2185,c2186,c2187,c2188,c2193,c2194,c2195,d1025,d1028,c2205
COMMON="--gres-flags=disable-binding -c 16 --mem=96G -t 2:00:00 -x $X"
bash eval/sbatch_once.sh -J cutrain3-slow -p gpu,gpu-short --gres=gpu:1 $COMMON -o $L/cutrain3_slow_%j.out $R slow
for p in gpu gpu-short; do
    for g in h200 a100; do
        bash eval/sbatch_once.sh -J cutrain3-fast -p $p --gres=gpu:$g:1 $COMMON -o $L/cutrain3_fast_%j.out $R fast
    done
done

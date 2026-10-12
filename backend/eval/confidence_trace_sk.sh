#!/bin/bash
module load miniconda3/25.9.1 >/dev/null 2>&1; source activate ~/envs/arr
cd ~/arrangement_transformer/backend
C=/home/basu.ang/arrangement_transformer/checkpoints
python -u eval/confidence_trace.py --m2m /scratch/basu.ang/m2m --audio /scratch/basu.ang/complication.wav \
  --cache /scratch/basu.ang/conf_cache_sk --stage $1 \
  --ckpt v1=$C/s2p_v1/best.pt --ckpt v3=$C/s2p_v3/best.pt --ckpt sk_S2=$C/sk_S2/best.pt \
  --ckpt sk_S46=$C/sk_S46/best.pt --ckpt sk=$C/s2p_sk/best.pt

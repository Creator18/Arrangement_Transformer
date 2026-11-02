#!/bin/bash
# Random-note curriculum (app/ml/spectral/train_random.py).  MODE=evalset | producer | train
#   evalset   one CPU job: the fixed seeded test clips per level -> $R/evalset
#   producer  CPU job array on courses: random clips made on the fly -> $R/spool (reads $R/control.json)
#   train     the GPU job: consumes the spool, mixes in arrangement renders, moves up levels
R=${R:-/scratch/basu.ang/random_curriculum}
D=/scratch/basu.ang/datasets
SF2S="$D/OmegaGMGS2.sf2,$D/soundfonts_new/fluid/FluidR3 GM2-2.SF2,$D/soundfonts_new/MuseScore_General.sf2"
module load miniconda3/25.9.1; source activate ~/envs/arr
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}   # node-local (a shared cache segfaulted)
export PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval
cd ~/arrangement_transformer/backend
mkdir -p $R/spool
PRESETS=/scratch/basu.ang/random_curriculum/random_presets.json
case "$MODE" in
  calib)    python eval/random_calib.py --sf2s "$SF2S" --sf-table $D/sf_table.json --out $PRESETS ;;
  evalset)  python eval/random_producer.py --sf2s "$SF2S" --sf-table $D/sf_table.json --evalset $R/evalset --per-level ${PER_LEVEL:-48} --presets $PRESETS ;;
  producer) python eval/random_producer.py --sf2s "$SF2S" --sf-table $D/sf_table.json --spool $R/spool --control $R/control.json \
                --max-minutes ${MINUTES:-470} --presets $PRESETS ;;
  train)    echo "$(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
            python -m app.ml.spectral.train_random --init ~/arrangement_transformer/checkpoints/base20m/best.pt \
                --data /scratch/basu.ang/base_render --spool $R/spool --control $R/control.json --evalset $R/evalset \
                --out ${OUT:-~/arrangement_transformer/checkpoints/arr_random} --max-minutes ${MINUTES:-450} $EXTRA ;;
esac 2>&1 | grep --line-buffered -v -i "warn\|pkg_resources"

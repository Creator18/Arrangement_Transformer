#!/bin/bash
# The realistic-singer batch, after the sung contents (eval/sing_batch.py) and the VocalSet voice
# model exist:  A2 clone every voice on 3 GPUs  ->  B render (levels + exposed voice + articulation
# both ways + cloned singers)  ->  C fine-tune LoRA adapters on arr_tone_full (gpu_train.sh: any GPU,
# moving to an A100 / H200 when one frees up)  ->  real-song comparison when it finishes.
cd ~/arrangement_transformer/backend
module load miniconda3/25.9.1; source activate ~/envs/arr
S=/scratch/basu.ang; C=~/arrangement_transformer/checkpoints; L=~/arrangement_transformer/logs; V=$S/voices
wait_jobs() { while squeue -u $USER -h -o %i | grep -q -E "^($(echo $@ | tr ' ' '|'))(_|$)"; do sleep 120; done; }
# A2: clone
python - <<PY
import json, glob
jobs = [j for f in sorted(glob.glob("$V/jobs_*.json")) for j in json.load(open(f))]
for k in range(3):
    json.dump(jobs[k::3], open(f"$V/rvc_part{k}.json", "w"))
print(len(jobs), "voices to clone")
PY
ids=()
for k in 0 1 2; do
  ids+=($(sbatch --parsable -p gpu --gres=gpu:1 -c 4 --mem=16G --time=08:00:00 -J sing-clone -o $L/singclone_${k}_%j.out \
    --wrap "module load FFmpeg/7.1.1; cd ~/tools/rvc && ~/envs/rvc/bin/python -u ~/arrangement_transformer/backend/eval/rvc_sing.py --model vocalset.pth --jobs $V/rvc_part$k.json"))
done
echo "clone jobs ${ids[@]}"; wait_jobs ${ids[@]}
echo "voices cloned: $(ls $V/*.voice.wav | wc -l)"
# B: render
R=$(sbatch --parsable -p short -c 1 --mem=8G --time=08:00:00 --array=0-149 -J singrender -o $L/singrender_%A_%a.out --wrap \
  "module load miniconda3/25.9.1; source activate ~/envs/arr; export NUMBA_CACHE_DIR=/tmp/nb_\$SLURM_JOB_ID_\$SLURM_ARRAY_TASK_ID; cd ~/arrangement_transformer/backend; python -u eval/render_arrangements.py --pop909 $S/datasets/pop909_tmp/POP909 --m2m $S/m2m --units $S/m2m/m2m_units.json --sf2 $S/datasets/OmegaGMGS2.sf2 --out $S/arr_sing --levels --art-both --voices $V --exposed-p 0.4 --multi-only --multi 6 --vocal-p 0.8 --slide-p 0.6 --tag s --shard \$SLURM_ARRAY_TASK_ID/150")
echo "render array $R"; wait_jobs $R
# C: train (new singer batch + the level batch, plus half as much of the earlier mix)
python -u eval/prep_ft_mix.py --new $S/arr_sing,$S/arr_level --old $S/ft_tone --out $S/ft_sing --old-frac 0.5
CPUS=12 bash eval/gpu_train.sh launch ft-sing $C/arr_sing_lora $L python -u -m app.ml.spectral.ft_aux --data $S/ft_sing \
  --base $C/arr_tone_full/best.pt --ckpt-dir $C/arr_sing_lora --mode lora --lora 16 --aux-weight 0.2 \
  --batch 16 --lr 4e-4 --total-steps 20000 --val-every 2000
sbatch -p short -c 1 --mem=1G --time=2-00:00:00 -J wait-sing -o $L/wait_sing_%j.out --wrap \
  "bash ~/arrangement_transformer/backend/eval/eval_when_done.sh $C/arr_sing_lora sing arr_tone_full"

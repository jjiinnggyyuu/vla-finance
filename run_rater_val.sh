#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh; conda activate vla-finance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
declare -A STEP=( [1]=2000 [2]=1000 [3]=1000 [4]=8000 [5]=2000 [6]=1000 )
run(){ local f=$1 g=$2 s=${STEP[$1]}
  echo "[$(date)] val-rater fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/dump_rater_val.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold${f}.yaml \
    --ckpt playground/Checkpoints/v15_btc_1h/fold${f}/checkpoints/steps_${s}_action_model.pt \
    --rater_pt results/exp4_rater/v15_btc_fold${f}.pt \
    --output_npz results/exp4_rater/v15_btc_fold${f}_val.npz --draws 20 \
    > logs/rater_val_fold${f}.log 2>&1
  echo "[$(date)] fold${f} done"; }
run 1 4 & run 2 5 & run 3 6 & run 4 7 & wait
run 5 4 & run 6 5 & wait
echo "[$(date)] ALL VAL-RATER DONE"

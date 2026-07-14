#!/bin/bash
# Unified evaluation: v14_multi_readoff + v16 MSAT (3 tracks), 6 folds each = 24 evals.
# Run AFTER run_v16_msat.sh finishes (needs the steps_10000 checkpoints).
# select_and_test.py picks best ckpt by val (da_k6, tiebreak aer_k6), tests, writes JSON
# with all k / strategies / slippage(0.1%) metrics. Plain python (no distributed / port).
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

ev () {  # exp fold gpu
  local exp=$1 fold=$2 gpu=$3
  mkdir -p results/${exp}
  echo "[$(date)] EVAL ${exp}/fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/${exp}/fold${fold}.yaml \
    --ckpt_dir    playground/Checkpoints/${exp}/fold${fold}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/${exp}/fold${fold}.json \
    > logs/eval_${exp}_fold${fold}.log 2>&1
  echo "[$(date)] DONE ${exp}/fold${fold}"
}

# Wave 1: v14_multi_readoff folds 1-4
ev v14_multi_readoff 1 4 & ev v14_multi_readoff 2 5 & ev v14_multi_readoff 3 6 & ev v14_multi_readoff 4 7 & wait
# Wave 2: v14_readoff 5-6 + v16_msat_btc 1-2
ev v14_multi_readoff 5 4 & ev v14_multi_readoff 6 5 & ev v16_msat_btc 1 6 & ev v16_msat_btc 2 7 & wait
# Wave 3: v16_msat_btc folds 3-6
ev v16_msat_btc 3 4 & ev v16_msat_btc 4 5 & ev v16_msat_btc 5 6 & ev v16_msat_btc 6 7 & wait
# Wave 4: v16_msat (multi) folds 1-4
ev v16_msat 1 4 & ev v16_msat 2 5 & ev v16_msat 3 6 & ev v16_msat 4 7 & wait
# Wave 5: v16_msat 5-6 + v16_msat_v14 1-2
ev v16_msat 5 4 & ev v16_msat 6 5 & ev v16_msat_v14 1 6 & ev v16_msat_v14 2 7 & wait
# Wave 6: v16_msat_v14 folds 3-6
ev v16_msat_v14 3 4 & ev v16_msat_v14 4 5 & ev v16_msat_v14 5 6 & ev v16_msat_v14 6 7 & wait

echo "[$(date)] ALL EVAL DONE"

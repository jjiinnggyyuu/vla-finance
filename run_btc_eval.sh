#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/v16_msat_btc
ev () {
  local fold=$1 gpu=$2
  echo "[$(date)] EVAL fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_btc/fold${fold}.yaml \
    --ckpt_dir    playground/Checkpoints/v16_msat_btc/fold${fold}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/v16_msat_btc/fold${fold}.json \
    > logs/eval_v16_msat_btc_fold${fold}.log 2>&1
  echo "[$(date)] EVAL DONE fold${fold}"
}
ev 1 4 & ev 2 5 & ev 3 6 & ev 4 7 & wait
ev 5 4 & ev 6 5 & wait
echo "[$(date)] === v16_msat_btc EVAL DONE ==="

#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results/v14_reeval

run_eval () {  # fold gpu
  local fold=$1 gpu=$2
  echo "[$(date)] EVAL v14/fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v14_groot_1h/starvla_train_bitcoin_v14_groot_1h_fold${fold}.yaml \
    --ckpt_dir playground/Checkpoints/v14_groot_1h/fold${fold}/checkpoints \
    --batch_size 2 \
    --output_json results/v14_reeval/fold${fold}.json \
    > logs/v14_reeval_fold${fold}.log 2>&1
  echo "[$(date)] DONE v14/fold${fold}"
}

run_eval 1 4 & run_eval 2 5 & run_eval 3 6 & run_eval 4 7 & wait
run_eval 5 4 & run_eval 6 5 & wait
echo "[$(date)] ALL V14 REEVAL DONE"

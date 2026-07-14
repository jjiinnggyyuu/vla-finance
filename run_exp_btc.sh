#!/bin/bash
# Phase 1 — v16_msat_btc: finish f5 (9000->10k) & f6 (8000->10k) via resume,
# then evaluate all 6 folds. Results -> results/v16_msat_btc/foldN.json
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs results/v16_msat_btc

echo "[disk] $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G free"

train () {  # fold gpu port resume
  local fold=$1 gpu=$2 port=$3 resume=$4
  echo "[$(date)] TRAIN v16_msat_btc/fold${fold} GPU${gpu} resume=${resume}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_btc/fold${fold}.yaml \
    --trainer.is_resume ${resume} > logs/v16_msat_btc_fold${fold}.log 2>&1
  echo "[$(date)] TRAIN DONE fold${fold}"
}
ev () {  # fold gpu
  local fold=$1 gpu=$2
  echo "[$(date)] EVAL v16_msat_btc/fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_btc/fold${fold}.yaml \
    --ckpt_dir    playground/Checkpoints/v16_msat_btc/fold${fold}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/v16_msat_btc/fold${fold}.json \
    > logs/eval_v16_msat_btc_fold${fold}.log 2>&1
  echo "[$(date)] EVAL DONE fold${fold}"
}

# --- finish remaining folds (resume) ---
train 5 4 29561 true & train 6 5 29562 true & wait

# --- evaluate all 6 folds ---
ev 1 4 & ev 2 5 & ev 3 6 & ev 4 7 & wait
ev 5 4 & ev 6 5 & wait

echo "[$(date)] === v16_msat_btc TRAIN+EVAL DONE ==="

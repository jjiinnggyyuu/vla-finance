#!/bin/bash
# Phase 2 — v16_msat (multi BTC+ETH+XRP): resume f1,f2 (5000->10k), fresh f3-6,
# then evaluate all 6 folds. Results -> results/v16_msat/foldN.json
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs results/v16_msat

echo "[disk] $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G free"

train () {  # fold gpu port resume
  local fold=$1 gpu=$2 port=$3 resume=$4
  echo "[$(date)] TRAIN v16_msat/fold${fold} GPU${gpu} resume=${resume}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat/fold${fold}.yaml \
    --trainer.is_resume ${resume} > logs/v16_msat_fold${fold}.log 2>&1
  echo "[$(date)] TRAIN DONE fold${fold}"
}
ev () {  # fold gpu
  local fold=$1 gpu=$2
  echo "[$(date)] EVAL v16_msat/fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat/fold${fold}.yaml \
    --ckpt_dir    playground/Checkpoints/v16_msat/fold${fold}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/v16_msat/fold${fold}.json \
    > logs/eval_v16_msat_fold${fold}.log 2>&1
  echo "[$(date)] EVAL DONE fold${fold}"
}

# --- train: resume f1,f2 + fresh f3-6 ---
train 1 4 29561 true  & train 2 5 29562 true  & \
train 3 6 29563 false & train 4 7 29564 false & wait
train 5 4 29561 false & train 6 5 29562 false & wait

# --- evaluate all 6 folds ---
ev 1 4 & ev 2 5 & ev 3 6 & ev 4 7 & wait
ev 5 4 & ev 6 5 & wait

echo "[$(date)] === v16_msat (multi) TRAIN+EVAL DONE ==="

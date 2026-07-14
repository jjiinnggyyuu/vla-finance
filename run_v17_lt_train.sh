#!/bin/bash
# v17 — integrated loss-prediction module ("13th-token", Learning Loss 1905.03677).
# Same DiT+BTC recipe as v15, but framework.action_model.loss_token=true adds a
# small head that predicts the chunk's flow-matching velocity loss (pairwise
# RANKING objective, target detached → price predictor/backbone protected).
# 6 folds, from-scratch JOINT training. Checkpoints → v17_btc_lt/fold{f}.
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs

train () {
  local fold=$1 gpu=$2 port=$3
  local tag="v17_btc_lt_fold${fold}"
  echo "[$(date)] START $tag GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold${fold}.yaml \
    --trainer.is_resume false \
    --framework.action_model.loss_token true \
    --framework.action_model.loss_token_weight 0.1 \
    --framework.action_model.loss_token_margin 1.0 \
    --framework.action_model.loss_token_detach true \
    --run_id v17_btc_lt/fold${fold} \
    > logs/${tag}.log 2>&1
  echo "[$(date)] DONE $tag"
}

# GPUs 4-7, two waves.
train 1 4 29571 & train 2 5 29572 & train 3 6 29573 & train 4 7 29574 & wait
train 5 4 29571 & train 6 5 29572 & wait
echo "[$(date)] ALL v17_lt TRAINING DONE"

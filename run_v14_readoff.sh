#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs
train () {
  local fold=$1 gpu=$2 port=$3
  echo "[$(date)] START fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v14_multi_readoff/fold${fold}.yaml \
    --trainer.is_resume false > logs/v14ro_fold${fold}.log 2>&1
  echo "[$(date)] DONE fold${fold}"
}
train 1 4 29531 & train 2 5 29532 & train 3 6 29533 & train 4 7 29534 & wait
train 5 4 29531 & train 6 5 29532 & wait
echo "[$(date)] ALL V14-READOFF DONE"

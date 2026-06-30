#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs

run_fold () {
  local fold=$1 gpu=$2 port=$3
  echo "[$(date)] START fold${fold} on GPU${gpu} port${port}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 \
    --main_process_port $port \
    starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v15_multi_1h/fold${fold}.yaml \
    --trainer.is_resume false \
    > logs/v15_multi_fold${fold}.log 2>&1
  echo "[$(date)] DONE fold${fold}"
}

# 1차: fold 1~4 → GPU 4,5,6,7 (포트 29501~29504)
run_fold 1 4 29501 & run_fold 2 5 29502 & run_fold 3 6 29503 & run_fold 4 7 29504 & wait
# 2차: fold 5~6 → GPU 4,5
run_fold 5 4 29501 & run_fold 6 5 29502 & wait
echo "[$(date)] ALL FOLDS DONE"

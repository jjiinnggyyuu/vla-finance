#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs

run_train () {
  local exp=$1 fold=$2 gpu=$3 port=$4
  echo "[$(date)] START ${exp}/fold${fold} on GPU${gpu} port${port}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 \
    --main_process_port $port \
    starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/${exp}/fold${fold}.yaml \
    --trainer.is_resume false \
    > logs/macro_${exp}_fold${fold}.log 2>&1
  echo "[$(date)] DONE ${exp}/fold${fold}"
}

run_train macro_baseline    1 4 29511 &
run_train macro_baseline    2 5 29512 &
run_train macro_btc_tlt_spy 1 6 29513 &
run_train macro_btc_tlt_spy 2 7 29514 &
wait
echo "[$(date)] ALL MACRO TRAIN DONE"

#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
echo "[$(date)] fold5 resume start (from steps_8000)"
CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 --main_process_port 29571 starVLA/training/train_starvla.py \
  --config_yaml examples/Bitcoin/train_files/v16_msat_btc/fold5.yaml \
  --trainer.is_resume true > logs/v16_msat_btc_fold5.log 2>&1
echo "[$(date)] fold5 DONE"

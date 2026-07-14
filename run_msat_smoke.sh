#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 --main_process_port 29541 starVLA/training/train_starvla.py \
  --config_yaml examples/Bitcoin/train_files/v16_msat_btc/fold1.yaml \
  --trainer.is_resume false \
  --trainer.max_train_steps 30 \
  --trainer.num_warmup_steps 5 \
  --trainer.save_interval 30 \
  --trainer.eval_interval 1000 > logs/msat_smoke_fold1.log 2>&1
echo "SMOKE_EXIT=$?"

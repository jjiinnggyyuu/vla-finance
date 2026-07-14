#!/bin/bash
# Missing-cell experiment: MSAT + normalisation OFF + multi all-diffusion (no read-off).
# Isolates normalisation's effect on the all-diffusion multi-asset setup vs v16_msat (norm ON).
# Train 6 folds fresh, then evaluate. Results -> results/v16_msat_raw/foldN.json
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs results/v16_msat_raw
echo "[disk] $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G free"

train () {  # fold gpu port
  local f=$1 g=$2 p=$3
  echo "[$(date)] TRAIN v16_msat_raw/fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $p starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_raw/fold${f}.yaml \
    --trainer.is_resume false > logs/v16_msat_raw_fold${f}.log 2>&1
  echo "[$(date)] TRAIN DONE fold${f}"
}
ev () {  # fold gpu
  local f=$1 g=$2
  echo "[$(date)] EVAL v16_msat_raw/fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_raw/fold${f}.yaml \
    --ckpt_dir    playground/Checkpoints/v16_msat_raw/fold${f}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/v16_msat_raw/fold${f}.json > logs/eval_v16_msat_raw_fold${f}.log 2>&1
  echo "[$(date)] EVAL DONE fold${f}"
}

# train (2 waves)
train 1 4 29591 & train 2 5 29592 & train 3 6 29593 & train 4 7 29594 & wait
train 5 4 29591 & train 6 5 29592 & wait
# eval (2 waves)
ev 1 4 & ev 2 5 & ev 3 6 & ev 4 7 & wait
ev 5 4 & ev 6 5 & wait

echo "[$(date)] === v16_msat_raw TRAIN+EVAL DONE ==="

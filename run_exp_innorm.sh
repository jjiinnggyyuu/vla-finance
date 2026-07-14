#!/bin/bash
# STEP 3: input normalisation on the best setup (v16_msat: MSAT+multi+all-diff+norm).
# Same as v16_msat but input_relative=true (candles as % vs current price).
# Tests whether scale-invariant input improves generalisation (esp. fold6 distribution shift).
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs results/v16_msat_innorm
echo "[disk] $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G free"

train () { local f=$1 g=$2 p=$3
  echo "[$(date)] TRAIN v16_msat_innorm/fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $p starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_innorm/fold${f}.yaml \
    --trainer.is_resume false > logs/v16_msat_innorm_fold${f}.log 2>&1
  echo "[$(date)] TRAIN DONE fold${f}"; }
ev () { local f=$1 g=$2
  echo "[$(date)] EVAL v16_msat_innorm/fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_innorm/fold${f}.yaml \
    --ckpt_dir playground/Checkpoints/v16_msat_innorm/fold${f}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/v16_msat_innorm/fold${f}.json > logs/eval_v16_msat_innorm_fold${f}.log 2>&1
  echo "[$(date)] EVAL DONE fold${f}"; }

train 1 4 29611 & train 2 5 29612 & train 3 6 29613 & train 4 7 29614 & wait
train 5 4 29611 & train 6 5 29612 & wait
ev 1 4 & ev 2 5 & ev 3 6 & ev 4 7 & wait
ev 5 4 & ev 6 5 & wait
echo "[$(date)] === v16_msat_innorm TRAIN+EVAL DONE ==="

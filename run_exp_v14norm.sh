#!/bin/bash
# Cell E: MSAT + read-off + normalisation ON. Fills the read-off vs all-diffusion
# comparison under normalisation (vs v16_msat), and normalisation effect on read-off (vs v16_msat_v14).
# Train 6 folds fresh, then evaluate. Results -> results/v16_msat_v14_norm/foldN.json
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs results/v16_msat_v14_norm
echo "[disk] $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G free"

train () {  # fold gpu port
  local f=$1 g=$2 p=$3
  echo "[$(date)] TRAIN v16_msat_v14_norm/fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $p starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_v14_norm/fold${f}.yaml \
    --trainer.is_resume false > logs/v16_msat_v14_norm_fold${f}.log 2>&1
  echo "[$(date)] TRAIN DONE fold${f}"
}
ev () {  # fold gpu
  local f=$1 g=$2
  echo "[$(date)] EVAL v16_msat_v14_norm/fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_v14_norm/fold${f}.yaml \
    --ckpt_dir    playground/Checkpoints/v16_msat_v14_norm/fold${f}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/v16_msat_v14_norm/fold${f}.json > logs/eval_v16_msat_v14_norm_fold${f}.log 2>&1
  echo "[$(date)] EVAL DONE fold${f}"
}

train 1 4 29601 & train 2 5 29602 & train 3 6 29603 & train 4 7 29604 & wait
train 5 4 29601 & train 6 5 29602 & wait
ev 1 4 & ev 2 5 & ev 3 6 & ev 4 7 & wait
ev 5 4 & ev 6 5 & wait

echo "[$(date)] === v16_msat_v14_norm TRAIN+EVAL DONE ==="

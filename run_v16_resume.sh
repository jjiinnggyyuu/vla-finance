#!/bin/bash
# Resume/complete v16 after the disk-full hang. Interrupted folds resume from their
# last checkpoint (is_resume true); not-started folds train fresh.
#
#   btc f5 (9000->10k) f6 (8000->10k)     : resume
#   multi f1,f2 (5000->10k)               : resume
#   multi f3-6, v14 f1-6                  : fresh
#
# Disk guard: abort a wave if free space < 15G (prevents another 100%-full hang).
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs

disk_guard () {
  local avail_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  echo "[disk] ${avail_g}G free"
  if [ "${avail_g:-0}" -lt 15 ]; then
    echo "[disk] ABORT: <15G free. Prune checkpoints before continuing."; exit 1
  fi
}

train () {  # cfgdir fold gpu port resume(true/false)
  local cfgdir=$1 fold=$2 gpu=$3 port=$4 resume=$5
  local tag="${cfgdir}_fold${fold}"
  echo "[$(date)] START $tag GPU${gpu} resume=${resume}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/${cfgdir}/fold${fold}.yaml \
    --trainer.is_resume ${resume} > logs/${tag}.log 2>&1
  echo "[$(date)] DONE $tag"
}

# Wave 1: complete interrupted folds (resume)
disk_guard
train v16_msat_btc 5 4 29561 true  & train v16_msat_btc 6 5 29562 true  & \
train v16_msat     1 6 29563 true  & train v16_msat     2 7 29564 true  & wait

# Wave 2: multi folds 3-6 (fresh)
disk_guard
train v16_msat 3 4 29561 false & train v16_msat 4 5 29562 false & \
train v16_msat 5 6 29563 false & train v16_msat 6 7 29564 false & wait

# Wave 3: v14-comparison folds 1-4 (fresh)
disk_guard
train v16_msat_v14 1 4 29561 false & train v16_msat_v14 2 5 29562 false & \
train v16_msat_v14 3 6 29563 false & train v16_msat_v14 4 7 29564 false & wait

# Wave 4: v14-comparison folds 5-6 (fresh)
disk_guard
train v16_msat_v14 5 4 29561 false & train v16_msat_v14 6 5 29562 false & wait

echo "[$(date)] ALL V16 RESUME DONE"

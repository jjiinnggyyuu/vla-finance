#!/bin/bash
# v16 MSAT experiments across GPUs 4-7 in waves of 4 (each fold ~10000 steps, ~1.4h).
#
#   Track            Norm  Head   Assets          DiT baseline to compare
#   ---------------  ----  -----  --------------  -----------------------
#   v16_msat_btc     ON    MSAT   BTC             v15_btc
#   v16_msat         ON    MSAT   BTC+ETH+XRP     v15_multi
#   v16_msat_v14     OFF   MSAT   BTC+ETH+XRP(RO) v14_multi_readoff
#
# 18 fold-runs total, 5 waves.
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs

train () {
  local cfgdir=$1 fold=$2 gpu=$3 port=$4
  local tag="${cfgdir}_fold${fold}"
  echo "[$(date)] START $tag GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/${cfgdir}/fold${fold}.yaml \
    --trainer.is_resume false > logs/${tag}.log 2>&1
  echo "[$(date)] DONE $tag"
}

# Wave 1: BTC-only (norm) folds 1-4
train v16_msat_btc 1 4 29561 & train v16_msat_btc 2 5 29562 & \
train v16_msat_btc 3 6 29563 & train v16_msat_btc 4 7 29564 & wait

# Wave 2: BTC-only folds 5-6 + multi (norm) folds 1-2
train v16_msat_btc 5 4 29561 & train v16_msat_btc 6 5 29562 & \
train v16_msat     1 6 29563 & train v16_msat     2 7 29564 & wait

# Wave 3: multi (norm) folds 3-6
train v16_msat 3 4 29561 & train v16_msat 4 5 29562 & \
train v16_msat 5 6 29563 & train v16_msat 6 7 29564 & wait

# Wave 4: v14-comparison (no-norm + readoff) folds 1-4
train v16_msat_v14 1 4 29561 & train v16_msat_v14 2 5 29562 & \
train v16_msat_v14 3 6 29563 & train v16_msat_v14 4 7 29564 & wait

# Wave 5: v14-comparison folds 5-6
train v16_msat_v14 5 4 29561 & train v16_msat_v14 6 5 29562 & wait

echo "[$(date)] ALL V16 MSAT DONE"

#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs

# 1) 현재 매크로 학습(1,2번)이 끝날 때까지 대기
echo "[$(date)] waiting for current macro train to finish..."
while pgrep -f "train_starvla.*macro_baseline\|train_starvla.*macro_btc_tlt_spy" >/dev/null; do sleep 60; done
echo "[$(date)] current macro done. starting next batch."

train () {  # exp fold gpu port
  local exp=$1 fold=$2 gpu=$3 port=$4
  echo "[$(date)] START ${exp}/fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port \
    starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/${exp}/fold${fold}.yaml \
    --trainer.is_resume false > logs/ov_${exp}_fold${fold}.log 2>&1
  echo "[$(date)] DONE ${exp}/fold${fold}"
}

# 2차: 매크로 IEF, SHY (4 run) — GPU 4,5,6,7
train macro_btc_spy_ief 1 4 29521 & train macro_btc_spy_ief 2 5 29522 & \
train macro_btc_spy_shy 1 6 29523 & train macro_btc_spy_shy 2 7 29524 & wait
echo "[$(date)] macro IEF/SHY done."

# 3차: v15_btc 6-fold — GPU 4,5,6,7 (4개씩 2배치)
train v15_btc_1h 1 4 29521 & train v15_btc_1h 2 5 29522 & \
train v15_btc_1h 3 6 29523 & train v15_btc_1h 4 7 29524 & wait
train v15_btc_1h 5 4 29521 & train v15_btc_1h 6 5 29522 & wait
echo "[$(date)] ALL OVERNIGHT DONE"

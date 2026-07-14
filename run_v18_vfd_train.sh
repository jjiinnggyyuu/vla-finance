#!/bin/bash
# v18 — VFD ensemble member 2 (Velocity-Field Disagreement, arXiv 2606.18043).
# Member 1 = existing v15 (seed 42). Member 2 = SAME recipe, seed 1, trained to
# the SAME per-fold step as v15's kept checkpoint (comparable sibling). VFD =
# velocity-field disagreement between the two heads during 4-step generation.
#
# Trains each fold ONLY to its target step and saves a single checkpoint there
# (fold4=8000, others 1000-2000) -> tiny disk footprint (~6 ckpts) + fast, so
# it won't re-fill the shared disk.
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs
# per-fold target step = v15 best-val step (member 1) so members are comparable
declare -A STEP=( [1]=2000 [2]=1000 [3]=1000 [4]=8000 [5]=2000 [6]=1000 )

train () {
  local fold=$1 gpu=$2 port=$3 s=${STEP[$1]}
  local tag="v18_vfd_m2_fold${fold}"
  echo "[$(date)] START $tag GPU${gpu} (train to ${s})"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $port starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold${fold}.yaml \
    --trainer.is_resume false \
    --seed 1 \
    --trainer.max_train_steps $s \
    --trainer.save_interval $s \
    --run_id v18_btc_vfd_m2/fold${fold} \
    > logs/${tag}.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date)] DONE $tag (rc=0)"; else echo "[$(date)] FAILED $tag (rc=$rc) <-- check log"; fi
}

# GPUs 4-7. fold4 (8000 steps) is the long pole (~1h); rest are quick.
train 1 4 29591 & train 2 5 29592 & train 3 6 29593 & train 4 7 29594 & wait
train 5 4 29591 & train 6 5 29592 & wait
echo "[$(date)] ALL v18_vfd member-2 TRAINING DONE"

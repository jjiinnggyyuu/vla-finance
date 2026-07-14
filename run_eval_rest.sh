#!/bin/bash
# Evaluate any missing multi / v14 folds across GPUs 4-7 (skips folds already done).
# Run AFTER killing run_v14_sched.sh (training is complete; only eval remains).
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results/v16_msat results/v16_msat_v14

evalfold () {  # exp fold gpu
  local e=$1 f=$2 g=$3
  if [ -f "results/${e}/fold${f}.json" ]; then echo "[skip] ${e} fold${f} (already done)"; return; fi
  echo "[$(date)] EVAL ${e} fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/${e}/fold${f}.yaml \
    --ckpt_dir    playground/Checkpoints/${e}/fold${f}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/${e}/fold${f}.json > logs/eval_${e}_fold${f}.log 2>&1
  echo "[$(date)] EVAL DONE ${e} fold${f}"
}

# All 12 evals; evalfold skips those already complete. 4 GPUs, waves of 4.
evalfold v16_msat     6 4 & evalfold v16_msat_v14 2 5 & evalfold v16_msat_v14 3 6 & evalfold v16_msat_v14 4 7 & wait
evalfold v16_msat_v14 5 4 & evalfold v16_msat_v14 6 5 & wait
# safety sweep: fill any still-missing (multi 1-5 / v14 1 normally already done)
evalfold v16_msat 1 4 & evalfold v16_msat 2 5 & evalfold v16_msat 3 6 & evalfold v16_msat 4 7 & wait
evalfold v16_msat 5 4 & evalfold v16_msat_v14 1 5 & wait

echo "[$(date)] === REMAINING EVAL DONE ==="

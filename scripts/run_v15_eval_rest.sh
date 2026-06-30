#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/v15_multi logs

run_eval () {
  local fold=$1 gpu=$2
  echo "[$(date)] EVAL fold${fold} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/v15_multi_1h/fold${fold}.yaml \
    --ckpt_dir playground/Checkpoints/v15_multi_1h/fold${fold}/checkpoints \
    --batch_size 2 \
    --output_json results/v15_multi/fold${fold}.json \
    > logs/v15_eval_fold${fold}.log 2>&1
  echo "[$(date)] DONE eval fold${fold}"
}

# fold 2,3,4,5 동시 (GPU 4,5,6,7) → fold 6 (GPU 4)
run_eval 2 4 & run_eval 3 5 & run_eval 4 6 & run_eval 5 7 & wait
run_eval 6 4 & wait
echo "[$(date)] ALL REST EVAL DONE"

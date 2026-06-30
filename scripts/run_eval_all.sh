#!/bin/bash
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results/macro results/v15_btc

run_eval () {  # exp fold gpu outdir
  local exp=$1 fold=$2 gpu=$3 outdir=$4
  echo "[$(date)] EVAL ${exp}/fold${fold} GPU${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/${exp}/fold${fold}.yaml \
    --ckpt_dir playground/Checkpoints/${exp}/fold${fold}/checkpoints \
    --batch_size 2 \
    --output_json results/${outdir}/${exp}_fold${fold}.json \
    > logs/eval_${exp}_fold${fold}.log 2>&1
  echo "[$(date)] DONE ${exp}/fold${fold}"
}

# 1차: 매크로 4종 fold1 (GPU 4,5,6,7)
run_eval macro_baseline     1 4 macro & run_eval macro_btc_tlt_spy 1 5 macro & \
run_eval macro_btc_spy_ief  1 6 macro & run_eval macro_btc_spy_shy 1 7 macro & wait
# 2차: 매크로 4종 fold2
run_eval macro_baseline     2 4 macro & run_eval macro_btc_tlt_spy 2 5 macro & \
run_eval macro_btc_spy_ief  2 6 macro & run_eval macro_btc_spy_shy 2 7 macro & wait
# 3차: v15_btc fold 1~4
run_eval v15_btc_1h 1 4 v15_btc & run_eval v15_btc_1h 2 5 v15_btc & \
run_eval v15_btc_1h 3 6 v15_btc & run_eval v15_btc_1h 4 7 v15_btc & wait
# 4차: v15_btc fold 5,6
run_eval v15_btc_1h 5 4 v15_btc & run_eval v15_btc_1h 6 5 v15_btc & wait
echo "[$(date)] ALL EVAL DONE"

#!/bin/bash
# Experiment 1: dump per-bar K-sample predictions for DiT+BTC (v15_btc_1h),
# all 6 folds, across the allocated GPUs 4-7. Each fold runs on one GPU
# (K-sampling is the compute, no need to split a fold across GPUs).
#
# Selected checkpoints come from each fold's val_ckpts.csv (selected=*).
# Output: results/exp1_samples/v15_btc_foldN.npz  (consumed by risk_coverage.py)
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/exp1_samples logs

K=${K:-10}          # noise draws per bar (override: K=20 ./run_dump_samples.sh)
CFG=examples/Bitcoin/train_files/v15_btc_1h
CKPT=playground/Checkpoints/v15_btc_1h
OUT=results/exp1_samples

# fold -> selected checkpoint step (from val_ckpts.csv)
declare -A STEP=( [1]=2000 [2]=1000 [3]=1000 [4]=8000 [5]=2000 [6]=1000 )

dump () {  # fold gpu
  local f=$1 g=$2 s=${STEP[$1]}
  echo "[$(date)] DUMP fold${f} GPU${g}  ckpt=steps_${s}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/dump_samples.py \
    --config_yaml ${CFG}/fold${f}.yaml \
    --ckpt ${CKPT}/fold${f}/checkpoints/steps_${s}_action_model.pt \
    --output_npz ${OUT}/v15_btc_fold${f}.npz \
    --num_samples ${K} --batch_size 8 \
    > logs/dump_v15_btc_fold${f}.log 2>&1
  echo "[$(date)] DONE fold${f}  (see logs/dump_v15_btc_fold${f}.log)"
}

# 6 folds over GPUs 4,5,6,7: first wave 4 folds, second wave 2 folds.
dump 1 4 & dump 2 5 & dump 3 6 & dump 4 7 & wait
dump 5 4 & dump 6 5 & wait

echo "[$(date)] === all folds dumped -> ${OUT}/ ==="
echo "Next: python -c \"import sys; sys.path.insert(0,'examples/Bitcoin/eval_files'); \\"
echo "  from risk_coverage import evaluate_strategies; \\"
echo "  evaluate_strategies([f'${OUT}/v15_btc_fold{i}.npz' for i in range(1,7)], k=6, theta=0.005)\""

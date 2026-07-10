#!/bin/bash
# Experiment 2: dump frozen VL features + predictions for DiT+BTC (v15_btc_1h),
# BOTH validation (train the confidence head) and test (evaluate) splits, all 6
# folds, across GPUs 4-7. VLM is run once per bar (feature) then the head is
# sampled K times, so this is much cheaper than exp1's dump.
#
# Output: results/exp2_features/v15_btc_fold{N}_{val,test}.npz
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/exp2_features logs

K=${K:-10}
CFG=examples/Bitcoin/train_files/v15_btc_1h
CKPT=playground/Checkpoints/v15_btc_1h
OUT=results/exp2_features
declare -A STEP=( [1]=2000 [2]=1000 [3]=1000 [4]=8000 [5]=2000 [6]=1000 )

feat () {  # fold gpu split
  local f=$1 g=$2 sp=$3 s=${STEP[$1]}
  echo "[$(date)] FEAT fold${f} ${sp} GPU${g} ckpt=steps_${s}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/dump_features.py \
    --config_yaml ${CFG}/fold${f}.yaml \
    --ckpt ${CKPT}/fold${f}/checkpoints/steps_${s}_action_model.pt \
    --split ${sp} \
    --output_npz ${OUT}/v15_btc_fold${f}_${sp:0:4}.npz \
    --num_samples ${K} --batch_size 8 \
    > logs/feat_v15_btc_fold${f}_${sp}.log 2>&1
  echo "[$(date)] DONE fold${f} ${sp}"
}

# Each fold does both splits sequentially on its GPU; 6 folds over GPUs 4-7.
run_fold () { feat $1 $2 validation; feat $1 $2 test; }
run_fold 1 4 & run_fold 2 5 & run_fold 3 6 & run_fold 4 7 & wait
run_fold 5 4 & run_fold 6 5 & wait

echo "[$(date)] === all features dumped -> ${OUT}/ ==="

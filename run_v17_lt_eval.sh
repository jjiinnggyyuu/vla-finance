#!/bin/bash
# v17 loss-token: dump per-bar confidence (val Nov + test Dec) then walk-forward eval.
# Reuses v15's pred/true features (results/exp2_features) — same as the exp4 rater
# eval — and only swaps in the integrated head's predicted loss.
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh; conda activate vla-finance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results/v17_lt
STEP=${STEP:-10000}          # checkpoint step to read the loss head from
DRAWS=${DRAWS:-20}

dump () {
  local f=$1 g=$2
  local ck=playground/Checkpoints/v17_btc_lt/fold${f}/checkpoints/steps_${STEP}_action_model.pt
  echo "[$(date)] dump fold${f} GPU${g} (ckpt steps_${STEP})"
  for split in validation test; do
    local suf=""; [ "$split" = validation ] && suf="_val"
    CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/dump_loss_token.py \
      --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold${f}.yaml \
      --ckpt "$ck" --split $split --draws $DRAWS \
      --output_npz results/v17_lt/v15_btc_fold${f}${suf}.npz \
      > logs/v17lt_dump_fold${f}_${split}.log 2>&1
  done
  echo "[$(date)] fold${f} dumped"
}

dump 1 4 & dump 2 5 & dump 3 6 & dump 4 7 & wait
dump 5 4 & dump 6 5 & wait
echo "[$(date)] ALL DUMPS DONE — running walk-forward eval"

for k in 12 6; do
  python examples/Bitcoin/eval_files/eval_rater_walkforward.py --k $k --rater_dir results/v17_lt
done

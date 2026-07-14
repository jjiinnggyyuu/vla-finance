#!/bin/bash
# v18 VFD: dump per-bar velocity-field disagreement (val Nov + test Dec) then
# walk-forward eval. Member 1 = v15 (best-val step per fold), member 2 = v18 seed1
# at the SAME step (comparable ensemble). Reuses v15 pred/true features.
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh; conda activate vla-finance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results/v18_vfd
DRAWS=${DRAWS:-8}
# v15 best-val checkpoint step per fold (member 1); member 2 uses the same step.
declare -A STEP=( [1]=2000 [2]=1000 [3]=1000 [4]=8000 [5]=2000 [6]=1000 )

dump () {
  local f=$1 g=$2 s=${STEP[$1]}
  local c1=playground/Checkpoints/v15_btc_1h/fold${f}/checkpoints/steps_${s}_action_model.pt
  local c2=playground/Checkpoints/v18_btc_vfd_m2/fold${f}/checkpoints/steps_${s}_action_model.pt
  echo "[$(date)] VFD dump fold${f} GPU${g} (step ${s})"
  for split in validation test; do
    local suf=""; [ "$split" = validation ] && suf="_val"
    CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/dump_vfd.py \
      --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold${f}.yaml \
      --ckpt1 "$c1" --ckpt2 "$c2" --split $split --draws $DRAWS \
      --output_npz results/v18_vfd/v15_btc_fold${f}${suf}.npz \
      > logs/v18vfd_dump_fold${f}_${split}.log 2>&1
  done
  echo "[$(date)] fold${f} dumped"
}

dump 1 4 & dump 2 5 & dump 3 6 & dump 4 7 & wait
dump 5 4 & dump 6 5 & wait
echo "[$(date)] ALL VFD DUMPS DONE — walk-forward eval + diagnostics"

for k in 12 6; do
  python examples/Bitcoin/eval_files/eval_rater_walkforward.py --k $k --rater_dir results/v18_vfd
done

# diagnostic: does VFD track realized error, and is it entangled with move size (exp3 trap)?
python - <<'PY'
import numpy as np
FEAT="results/exp2_features"; VF="results/v18_vfd"
cs=[]; ms=[]
print("fold | corr(VFD,realized_err)  corr(VFD,|move|)")
for f in range(1,7):
    fe=np.load(f"{FEAT}/v15_btc_fold{f}_test.npz"); vf=np.load(f"{VF}/v15_btc_fold{f}.npz")
    pred=fe["sample_rets_k12"][:,0]; true=fe["true_ret_k12"]; conf=vf["pred_logloss"][:,11]
    err=(pred-true)**2; move=np.abs(true)
    c=np.corrcoef(conf,err)[0,1]; m=np.corrcoef(conf,move)[0,1]; cs.append(c); ms.append(m)
    print(f"  {f}  |   {c:+.3f}                {m:+.3f}")
print(f"\nmean corr(VFD, realized_err) = {np.mean(cs):+.3f}")
print(f"mean corr(VFD, |move|)       = {np.mean(ms):+.3f}  (low = avoids boring-bar trap)")
PY

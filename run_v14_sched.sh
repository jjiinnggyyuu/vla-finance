#!/bin/bash
# Dynamic scheduler: train v16_msat_v14 (6 folds) using idle GPUs 6,7 now, then
# 4,5 once multi's orphan f5,f6 free them. Finally, unified eval of multi(6) + v14(6).
set -uo pipefail
cd /home/jingyu/vla-finance
source /home/jingyu/miniconda3/etc/profile.d/conda.sh
conda activate vla-finance
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p logs results/v16_msat_v14 results/v16_msat

trainv14 () {  # fold gpu port
  local f=$1 g=$2 p=$3
  echo "[$(date)] TRAIN v14 fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 --main_process_port $p starVLA/training/train_starvla.py \
    --config_yaml examples/Bitcoin/train_files/v16_msat_v14/fold${f}.yaml \
    --trainer.is_resume false > logs/v16_msat_v14_fold${f}.log 2>&1
  echo "[$(date)] TRAIN DONE v14 fold${f}"
}
evalfold () {  # exp fold gpu
  local e=$1 f=$2 g=$3
  echo "[$(date)] EVAL ${e} fold${f} GPU${g}"
  CUDA_VISIBLE_DEVICES=$g python examples/Bitcoin/eval_files/select_and_test.py \
    --config_yaml examples/Bitcoin/train_files/${e}/fold${f}.yaml \
    --ckpt_dir    playground/Checkpoints/${e}/fold${f}/checkpoints \
    --batch_size 2 --slippage 0.001 \
    --output_json results/${e}/fold${f}.json > logs/eval_${e}_fold${f}.log 2>&1
  echo "[$(date)] EVAL DONE ${e} fold${f}"
}
gpu_free () {  # gpu -> true if <2GB used
  local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -dc '0-9')
  [ "${used:-99999}" -lt 2000 ]
}

# 1) start f1,f2 on the idle GPUs 6,7
trainv14 1 6 29581 & P1=$!
trainv14 2 7 29582 & P2=$!

# 2) wait until GPU4 AND GPU5 are free (multi orphan f5,f6 finished)
echo "[sched] waiting for GPU4,5 to free (multi f5,f6)..."
until gpu_free 4 && gpu_free 5; do sleep 60; done
sleep 20   # let GPU memory fully release
echo "[sched] GPU4,5 free -> launching v14 f3,f4"

# 3) f3,f4 on 4,5 (run concurrently with f1,f2 still on 6,7)
trainv14 3 4 29583 & P3=$!
trainv14 4 5 29584 & P4=$!

# 4) after the first four finish, run f5,f6
wait $P1 $P2 $P3 $P4
trainv14 5 6 29581 & trainv14 6 7 29582 & wait

# 5) unified eval: multi f1-6 + v14 f1-6 (all training now complete)
echo "[sched] all training done -> unified eval (multi + v14)"
evalfold v16_msat 1 4 & evalfold v16_msat 2 5 & evalfold v16_msat 3 6 & evalfold v16_msat 4 7 & wait
evalfold v16_msat 5 4 & evalfold v16_msat 6 5 & evalfold v16_msat_v14 1 6 & evalfold v16_msat_v14 2 7 & wait
evalfold v16_msat_v14 3 4 & evalfold v16_msat_v14 4 5 & evalfold v16_msat_v14 5 6 & evalfold v16_msat_v14 6 7 & wait

echo "[$(date)] === ALL DONE: multi + v14 train+eval ==="

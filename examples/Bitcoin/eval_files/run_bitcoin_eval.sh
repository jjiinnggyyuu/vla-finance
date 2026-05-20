#!/usr/bin/env bash
set -euo pipefail

config_yaml=${config_yaml:-./examples/Bitcoin/train_files/starvla_train_bitcoin.yaml}
ckpt=${ckpt:?Set ckpt=/path/to/steps_xxx_pytorch_model.pt}
split=${split:-validation}
batch_size=${batch_size:-1}
num_workers=${num_workers:-2}
output_json=${output_json:-./playground/Checkpoints/bitcoin_eval_${split}_close_mae.json}

python examples/Bitcoin/eval_files/eval_bitcoin_close_mae.py \
  --config_yaml "${config_yaml}" \
  --ckpt "${ckpt}" \
  --split "${split}" \
  --batch_size "${batch_size}" \
  --num_workers "${num_workers}" \
  --output_json "${output_json}"

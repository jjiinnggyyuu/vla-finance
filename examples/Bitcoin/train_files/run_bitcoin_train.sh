#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

Framework_name=${Framework_name:-QwenGR00T}
freeze_module_list=${freeze_module_list:-qwen_vl_interface}
base_vlm=${base_vlm:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/Bitcoin/train_files/starvla_train_bitcoin.yaml}
bitcoin_data_root=${bitcoin_data_root:-.}
bitcoin_csv_file=${bitcoin_csv_file:-btc_1h.csv}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-bitcoin_qwengroot_ohlc12}
num_processes=${NUM_PROCESSES:-4}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${bitcoin_data_root}" \
  --datasets.vla_data.csv_file "${bitcoin_csv_file}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --trainer.is_resume true

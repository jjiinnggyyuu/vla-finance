#!/usr/bin/env bash
# Evaluate all checkpoints in parallel across GPUs 4,5,6,7
set -euo pipefail

GPUS=(4 5 6 7)

config_yaml=${config_yaml:-./examples/Bitcoin/train_files/starvla_train_bitcoin.yaml}
ckpt_dir=${ckpt_dir:-./playground/Checkpoints/bitcoin_text_v2/checkpoints}
split=${split:-validation}
batch_size=${batch_size:-4}
num_workers=${num_workers:-2}
output_dir=${output_dir:-./playground/Checkpoints/bitcoin_text_v2/eval_results}

mkdir -p "${output_dir}"

echo "=============================================="
echo " Evaluating all checkpoints on: ${split}"
echo " (parallel across GPUs: ${GPUS[*]})"
echo "=============================================="

pids=()
gpu_idx=0

for ckpt_path in "${ckpt_dir}"/steps_*_action_model.pt; do
    ckpt_name=$(basename "${ckpt_path}" .pt)
    output_json="${output_dir}/${ckpt_name}_${split}.json"
    gpu=${GPUS[$gpu_idx]}

    echo ">>> ${ckpt_name}  (GPU ${gpu})"

    CUDA_VISIBLE_DEVICES=${gpu} python examples/Bitcoin/eval_files/eval_bitcoin_close_mae.py \
        --config_yaml "${config_yaml}" \
        --ckpt "${ckpt_path}" \
        --split "${split}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --output_json "${output_json}" &

    pids+=($!)
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))

    # 4개씩 묶어서 완료 대기 (GPU 4개 동시 실행)
    if (( ${#pids[@]} % ${#GPUS[@]} == 0 )); then
        for pid in "${pids[@]}"; do wait "${pid}"; done
        pids=()
    fi
done

# 나머지 대기
for pid in "${pids[@]}"; do wait "${pid}"; done

echo ""
echo "========================================================================================================"
echo " SUMMARY"
echo "========================================================================================================"
for json_file in "${output_dir}"/steps_*_${split}.json; do
    ckpt_name=$(basename "${json_file}" _${split}.json)
    python -c "
import json
d = json.load(open('${json_file}'))
name = '${ckpt_name}'
print(f'  {name}')
print(f'    MAE={d[\"close_mae\"]:.2f}  IC={d[\"price_ic\"]:.6f}  RankIC={d[\"price_rankic\"]:.6f}')
print(f'    {\"\":12} {\"k=1\":>8} {\"k=3\":>8} {\"k=6\":>8} {\"k=12\":>8}')
print(f'    {\"DA\":12} {d.get(\"da_k1\",0):>8.4f} {d.get(\"da_k3\",0):>8.4f} {d.get(\"da_k6\",0):>8.4f} {d.get(\"da_k12\",0):>8.4f}')
print(f'    {\"AER\":12} {d.get(\"aer_k1\",0):>8.4f} {d.get(\"aer_k3\",0):>8.4f} {d.get(\"aer_k6\",0):>8.4f} {d.get(\"aer_k12\",0):>8.4f}')
print(f'    {\"Sortino\":12} {d.get(\"sortino_k1\",0):>8.4f} {d.get(\"sortino_k3\",0):>8.4f} {d.get(\"sortino_k6\",0):>8.4f} {d.get(\"sortino_k12\",0):>8.4f}')
print(f'    {\"MDD\":12} {d.get(\"mdd_k1\",0)*100:>7.2f}% {d.get(\"mdd_k3\",0)*100:>7.2f}% {d.get(\"mdd_k6\",0)*100:>7.2f}% {d.get(\"mdd_k12\",0)*100:>7.2f}%')
print()
"
done

echo "========================================================================================================"
echo "Results saved in: ${output_dir}"

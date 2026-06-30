#!/usr/bin/env bash
# Kronos 베이스라인 6-fold test 평가 (v15와 동일한 fold 구조).
# 각 fold의 test split으로 추론 → JSON 저장. 끝나면 aggregate로 평균.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

YAML_DIR="examples/Bitcoin/train_files/v15_dct_ablation"
MODEL_ID=${MODEL_ID:-NeoQuasar/Kronos-base}
OUT_DIR=${OUT_DIR:-results/kronos_baseline}
SAMPLE_COUNT=${SAMPLE_COUNT:-5}
DEVICE=${DEVICE:-cuda:0}

mkdir -p "${OUT_DIR}"

# fold별 config는 oft/oft_dct 어느 쪽이든 split 날짜가 같으므로 oft 하나만 사용
for fold in 1 2 3 4 5 6; do
    yaml="${YAML_DIR}/starvla_train_bitcoin_v15_fold${fold}_oft.yaml"
    echo "================================================================"
    echo "  Kronos fold${fold} test 평가  (model=${MODEL_ID})"
    echo "================================================================"
    python examples/Bitcoin/eval_files/eval_kronos_baseline.py \
        --config_yaml "${yaml}" \
        --split test \
        --sample_count "${SAMPLE_COUNT}" \
        --model_id "${MODEL_ID}" \
        --device "${DEVICE}" \
        --output_json "${OUT_DIR}/fold${fold}.json" \
        --plot_dir "${OUT_DIR}/fold${fold}/"
done

echo ""
echo "================================================================"
echo "  6-fold 평균 집계"
echo "================================================================"
python examples/Bitcoin/eval_files/aggregate_folds.py "${OUT_DIR}"/fold*.json

#!/usr/bin/env bash
# 범용 평가 스크립트 — fold별 model selection + test, (전체일 때) 6-fold 평균.
#
# 사용법:
#   bash run_eval.sh <yaml_dir> [out_label] [fold_filter]
#
#   <yaml_dir>   : 실험 폴더 (starvla_train_*.yaml)
#   [out_label]  : 결과 폴더명 results/<label> (기본: 폴더명)
#   [fold_filter]: 평가할 fold만 지정 (예: "fold1 fold2"). 주면 그 fold만 돌고
#                  집계는 건너뜀(부분 실행). 여러 GPU로 나눠 돌릴 때 사용.
#
# 예:
#   # 한 GPU로 전체 (자동 집계)
#   CUDA_VISIBLE_DEVICES=4 bash run_eval.sh .../v14_groot_1h v14_groot
#   # GPU 4~7로 나눠 동시 평가 → 마지막에 집계만 따로:
#   CUDA_VISIBLE_DEVICES=4 bash run_eval.sh .../v14_groot_1h v14_groot "fold1 fold2" &
#   CUDA_VISIBLE_DEVICES=5 bash run_eval.sh .../v14_groot_1h v14_groot "fold3 fold4" &
#   ...; wait; bash run_eval.sh .../v14_groot_1h v14_groot aggregate
set -euo pipefail

# conda 환경 보장 (tmux/nohup 등 비대화형 셸 대비)
if ! command -v python >/dev/null 2>&1 || ! python -c "import accelerate" >/dev/null 2>&1; then
    source /home/jingyu/miniconda3/etc/profile.d/conda.sh
    conda activate vla-finance
fi

YAML_DIR="${1:?Usage: run_eval.sh <yaml_dir> [out_label] [fold_filter] [num_processes]}"
LABEL="${2:-$(basename "${YAML_DIR}")}"
FOLD_FILTER="${3:-}"
NUM_PROCESSES="${4:-${NUM_PROCESSES:-1}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

CKPT_ROOT="playground/Checkpoints"
OUT_DIR="results/${LABEL}"
mkdir -p "${OUT_DIR}"

# fold_filter == "aggregate" 면 평가는 건너뛰고 집계만 수행 (나눠 돌린 뒤 마무리용).
if [ "${FOLD_FILTER}" = "aggregate" ]; then
    echo "=== aggregate only: ${LABEL} ==="
    # run_id에 슬래시가 있어 fold JSON은 ${OUT_DIR}/<run_id앞>/foldN.json 처럼 중첩됨.
    # 재귀로 찾고 _avg 등 메타(_로 시작)는 제외.
    mapfile -t FOLD_JSONS < <(find "${OUT_DIR}" -name '*.json' ! -name '_*' | sort)
    python examples/Bitcoin/eval_files/aggregate_folds.py \
        "${FOLD_JSONS[@]}" --label "${LABEL}" \
        --output_json "${OUT_DIR}/_avg.json"
    exit 0
fi

mapfile -t ALL_YAMLS < <(ls "${YAML_DIR}"/starvla_train_*.yaml | sort)
# fold_filter 가 있으면 파일명에 토큰이 든 yaml만 선택.
if [ -n "${FOLD_FILTER}" ]; then
    YAMLS=()
    for y in "${ALL_YAMLS[@]}"; do
        for tok in ${FOLD_FILTER}; do
            if [[ "$(basename "$y")" == *"${tok}"* ]]; then YAMLS+=("$y"); break; fi
        done
    done
else
    YAMLS=("${ALL_YAMLS[@]}")
fi
TOTAL=${#YAMLS[@]}
if [ "${TOTAL}" -eq 0 ]; then
    echo "No matching starvla_train_*.yaml in ${YAML_DIR} (filter: '${FOLD_FILTER}')" >&2
    exit 1
fi

for i in "${!YAMLS[@]}"; do
    yaml="${YAMLS[$i]}"
    run_id=$(grep '^run_id:' "${yaml}" | awk '{print $2}')
    ckpt_dir="${CKPT_ROOT}/${run_id}/checkpoints"
    out_json="${OUT_DIR}/${run_id}.json"
    plot_dir="${OUT_DIR}/${run_id}/"

    echo "================================================================"
    echo "  [$(( i + 1 ))/${TOTAL}] Eval (select+test): ${run_id}"
    echo "================================================================"
    if [ ! -d "${ckpt_dir}" ]; then
        echo "  [skip] checkpoint dir not found: ${ckpt_dir}" >&2
        continue
    fi
    accelerate launch \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "${MAIN_PORT:-29500}" \
        examples/Bitcoin/eval_files/select_and_test.py \
        --config_yaml "${yaml}" \
        --ckpt_dir "${ckpt_dir}" \
        --output_json "${out_json}" \
        --plot_dir "${plot_dir}"
done

# fold_filter 가 지정된 부분 실행이면 집계 생략 (나눠 돌리는 중이므로).
# 전체 실행일 때만 자동 집계.
if [ -n "${FOLD_FILTER}" ]; then
    echo ""
    echo "부분 평가 완료 (filter: '${FOLD_FILTER}'). 집계는 모든 fold 끝난 뒤:"
    echo "  bash $0 ${YAML_DIR} ${LABEL} aggregate"
    exit 0
fi

echo ""
echo "================================================================"
echo "  ${TOTAL}-fold 평균: ${LABEL}"
echo "================================================================"
# fold 결과 JSON만 집계 (_avg.json 같은 메타 파일 제외).
mapfile -t FOLD_JSONS < <(ls "${OUT_DIR}"/*.json | grep -v '/_')
python examples/Bitcoin/eval_files/aggregate_folds.py \
    "${FOLD_JSONS[@]}" --label "${LABEL}" \
    --output_json "${OUT_DIR}/_avg.json"

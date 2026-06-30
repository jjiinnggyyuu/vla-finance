#!/usr/bin/env bash
# 범용 학습 스크립트 — 폴더 안의 fold YAML을 순차 학습.
#
# 사용법:
#   bash run_train.sh <yaml_dir> [num_processes] [fold_filter]
#
#   <yaml_dir>     : starvla_train_*.yaml 들이 있는 실험 폴더 (정본 하나)
#   [num_processes]: GPU 프로세스 수 (기본 4)
#   [fold_filter]  : 돌릴 fold만 공백 구분으로 지정 (예: "fold1 fold2 fold3").
#                    생략 시 폴더 안 모든 yaml. 병렬/재시도 시 복사폴더 만들지 말고 이걸로 분리.
#
# 예:
#   # 전체 순차
#   bash run_train.sh examples/Bitcoin/train_files/v14_groot_1h 2
#   # 병렬 (복사폴더 불필요): GPU 4,5 → fold1~3 / GPU 6,7 → fold4~6
#   CUDA_VISIBLE_DEVICES=4,5 MAIN_PORT=29500 bash run_train.sh <dir> 2 "fold1 fold2 fold3"
#   CUDA_VISIBLE_DEVICES=6,7 MAIN_PORT=29600 bash run_train.sh <dir> 2 "fold4 fold5 fold6"
#   # 재시도(터진 fold만)
#   CUDA_VISIBLE_DEVICES=6,7 MAIN_PORT=29600 bash run_train.sh <dir> 2 "fold6"
set -euo pipefail

# conda 환경 보장 (tmux/nohup 등 비대화형 셸에서 accelerate not found 방지)
if ! command -v accelerate >/dev/null 2>&1; then
    source /home/jingyu/miniconda3/etc/profile.d/conda.sh
    conda activate vla-finance
fi

YAML_DIR="${1:?Usage: run_train.sh <yaml_dir> [num_processes] [fold_filter]}"
NUM_PROCESSES="${2:-${NUM_PROCESSES:-4}}"
FOLD_FILTER="${3:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

mapfile -t ALL_YAMLS < <(ls "${YAML_DIR}"/starvla_train_*.yaml | sort)
# fold_filter 가 있으면 파일명에 해당 토큰이 들어간 yaml만 선택.
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

echo "Found ${TOTAL} configs in ${YAML_DIR} (GPUs: ${CUDA_VISIBLE_DEVICES}, procs: ${NUM_PROCESSES})"

for i in "${!YAMLS[@]}"; do
    yaml="${YAMLS[$i]}"
    run_id=$(grep '^run_id:' "${yaml}" | awk '{print $2}')
    echo "================================================================"
    echo "  [$(( i + 1 ))/${TOTAL}] Training: ${run_id}"
    echo "  Config: ${yaml}"
    echo "================================================================"
    accelerate launch \
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "${MAIN_PORT:-29500}" \
        starVLA/training/train_starvla.py \
        --config_yaml "${yaml}"
    echo "  [$(( i + 1 ))/${TOTAL}] Done: ${run_id}"
    echo ""
done

echo "All ${TOTAL} trainings complete."

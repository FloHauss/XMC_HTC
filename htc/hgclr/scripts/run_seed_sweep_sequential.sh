#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-contrastive-htc}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_ID="${GPU_ID:-0}"
TRAIN_BATCH="${TRAIN_BATCH:-12}"
EVAL_BATCH="${EVAL_BATCH:-32}"
SEEDS=(${SEEDS:-1 2 3 4 5})
DATASETS=(${DATASETS:-WebOfScience nyt rcv1})

declare -A LAMB=(
  [WebOfScience]=0.05
  [nyt]=0.3
  [rcv1]=0.3
)

declare -A THRE=(
  [WebOfScience]=0.02
  [nyt]=0.002
  [rcv1]=0.001
)

mkdir -p checkpoints

if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
fi

echo "Starting sequential HGCLR seed sweep on GPU ${GPU_ID}"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
if [[ -n "${PYTHON_BIN}" ]]; then
  echo "Python: ${PYTHON_BIN}"
else
  echo "Conda env: ${CONDA_ENV_NAME}"
fi

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_suffix="seed${seed}"
    run_name="${dataset}-${run_suffix}"
    run_dir="checkpoints/${run_name}"
    train_log="${run_dir}/train.stdout.log"
    eval_log="${run_dir}/test.stdout.log"

    mkdir -p "${run_dir}"

    echo
    echo "=== ${run_name}: training ==="
    if [[ -f "${run_dir}/checkpoint_best_macro.pt" && -f "${run_dir}/checkpoint_best_micro.pt" ]]; then
      echo "Skipping training for ${run_name}; checkpoints already exist."
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_CMD[@]}" train.py \
        --name "${run_suffix}" \
        --data "${dataset}" \
        --batch "${TRAIN_BATCH}" \
        --lamb "${LAMB[$dataset]}" \
        --thre "${THRE[$dataset]}" \
        --seed "${seed}" | tee "${train_log}"
    fi

    echo "=== ${run_name}: evaluation (_macro) ==="
    if grep -q '"test_metrics"' "${run_dir}/cost_metrics.json" 2>/dev/null; then
      echo "Skipping evaluation for ${run_name}; test metrics already recorded."
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_CMD[@]}" test.py \
        --name "${run_name}" \
        --batch "${EVAL_BATCH}" | tee "${eval_log}"
    fi
  done
done

echo
echo "Sweep complete. Building summary CSV."
"${PYTHON_CMD[@]}" scripts/summarize_seed_sweep.py
echo "Building per-dataset aggregate files."
"${PYTHON_CMD[@]}" scripts/aggregate_seed_results.py --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}"

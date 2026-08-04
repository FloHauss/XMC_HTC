#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(${DATASETS:-nyt rcv1})
SEEDS=(${SEEDS:-1 2 3 4 5})
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    job_name="hgclr-${dataset}-seed${seed}"
    sbatch_args=(--job-name="${job_name}")
    if [[ -n "${SBATCH_ACCOUNT}" ]]; then
      sbatch_args+=(--account="${SBATCH_ACCOUNT}")
    fi
    if [[ -n "${SBATCH_PARTITION}" ]]; then
      sbatch_args+=(--partition="${SBATCH_PARTITION}")
    fi
    sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/bwunicluster_single_run.sbatch" "${dataset}" "${seed}"
  done
done

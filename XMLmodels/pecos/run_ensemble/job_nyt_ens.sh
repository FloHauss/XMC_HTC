#!/bin/bash
#SBATCH --partition=gpu_4_a100
#SBATCH --time=5:00:00
#SBATCH --mem=40000
#SBATCH --job-name=nyt
#SBATCH --gres=gpu:2
#SBATCH --dependency=singleton

# [x] sbatch params optimized

DATASET="nyt"
PATH_TO_DATASET="htc-base"

NOW=$(date "+%Y-%m-%d %H:%M:%S")

if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi
conda activate xr_transformer_env

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "------- Ensemble run at $NOW for $DATASET ----------"

bash run.sh ${DATASET} ${PATH_TO_DATASET}

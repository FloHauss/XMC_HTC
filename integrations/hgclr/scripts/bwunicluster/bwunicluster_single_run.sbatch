#!/usr/bin/env bash
#SBATCH --output=checkpoints/slurm-%x-%j.out
#SBATCH --error=checkpoints/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --time=2-00:00:00

set -euo pipefail

dataset="${1:?Usage: sbatch bwunicluster_single_run.sbatch DATASET SEED}"
seed="${2:?Usage: sbatch bwunicluster_single_run.sbatch DATASET SEED}"

case "${dataset}" in
  nyt)
    lamb=0.3
    thre=0.002
    ;;
  rcv1)
    lamb=0.3
    thre=0.001
    ;;
  *)
    echo "Unsupported dataset for remaining sweep: ${dataset}" >&2
    exit 2
    ;;
esac

cd "${SLURM_SUBMIT_DIR}"
mkdir -p checkpoints

# Adjust this block to match the BwUniCluster software stack if conda is
# provided through modules rather than your login shell.
if command -v conda >/dev/null 2>&1; then
  :
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-contrastive-htc}"
TRAIN_BATCH="${TRAIN_BATCH:-12}"
EVAL_BATCH="${EVAL_BATCH:-32}"
run_suffix="seed${seed}"
run_name="${dataset}-${run_suffix}"
run_dir="checkpoints/${run_name}"

mkdir -p "${run_dir}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "dataset=${dataset}"
echo "seed=${seed}"
date -u

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
PY

if [[ -f "${run_dir}/checkpoint_best_macro.pt" && -f "${run_dir}/checkpoint_best_micro.pt" ]]; then
  echo "Skipping training for ${run_name}; checkpoints already exist."
else
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" python train.py \
    --name "${run_suffix}" \
    --data "${dataset}" \
    --batch "${TRAIN_BATCH}" \
    --lamb "${lamb}" \
    --thre "${thre}" \
    --seed "${seed}" | tee "${run_dir}/train.stdout.log"
fi

if grep -q '"test_metrics"' "${run_dir}/cost_metrics.json" 2>/dev/null; then
  echo "Skipping evaluation for ${run_name}; test metrics already recorded."
else
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" python test.py \
    --name "${run_name}" \
    --batch "${EVAL_BATCH}" | tee "${run_dir}/test.stdout.log"
fi

date -u
echo "Completed ${run_name}"

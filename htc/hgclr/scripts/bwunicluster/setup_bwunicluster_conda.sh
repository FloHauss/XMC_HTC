#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-contrastive-htc}"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "conda is not available. Load the BwUniCluster conda/miniforge module first, then rerun this script." >&2
    exit 1
  fi
fi

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "Conda env ${CONDA_ENV_NAME} already exists."
else
  conda create -n "${CONDA_ENV_NAME}" python=3.10 pip -y
fi

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m pip install \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m pip install \
  numpy==2.2.6 \
  transformers==4.30.2 \
  fairseq==0.10.0 \
  scikit-learn \
  tqdm

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m pip install \
  torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch
import transformers
from fairseq.data import data_utils
import sklearn
import tqdm

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("env_check_ok")
PY

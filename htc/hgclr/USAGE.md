# Usage Guide

This document covers setup and usage for the HGCLR integration used in the
cross-domain study. The original model and training code are described in
[UPSTREAM_README.md](UPSTREAM_README.md).

## Changes from the original

- Preprocessing scripts that read from pre-cleaned HTC datasets (no raw data download required)
- Extended evaluation: **P@1**, **P@3**, **P@5**, **R-Precision** in addition to Macro/Micro-F1
- Cost metric logging: training time, GPU memory, throughput, and inference time written to `cost_metrics.json`

---

## Environment

A dedicated conda environment is required. PyTorch 2.x is needed for CUDA sm_90+ GPUs (H100).

```bash
conda create -n contrastive-htc python=3.10 pip -y
conda activate contrastive-htc
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==2.2.6 transformers==4.30.2 fairseq==0.10.0 scikit-learn tqdm
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

---

## Data Preprocessing

The preprocessing scripts read cleaned JSONL datasets from an explicit input
directory and write generated binaries into the respective integration data
directory by default. The generated `.bin`, `.idx` and `.pt` files are ignored
by Git.

```bash
python data/nyt/preprocess_from_htc.py --input-dir /path/to/nyt
python data/rcv1/preprocess_from_htc.py --input-dir /path/to/rcv1
python data/WebOfScience/preprocess_from_htc.py --input-dir /path/to/wos
```

Each script fails if a dataset label is absent from the configured vocabulary or
taxonomy, or if any sample would have no recognised gold label.

---

## Training

```bash
python3 train.py --name NAME --data {WebOfScience,nyt,rcv1} [options]
```

The study used the following upstream reference hyperparameters for five runs
with seeds 1-5:

| Dataset      | `--lamb` | `--thre` |
|--------------|----------|----------|
| WebOfScience | 0.05     | 0.02     |
| NYT          | 0.3      | 0.002    |
| RCV1         | 0.3      | 0.001    |

Example:

```bash
python3 train.py --name run1 --data WebOfScience --batch 12 --lamb 0.05 --thre 0.02 --seed 3
```

Checkpoints are saved to `checkpoints/WebOfScience-run1/`.

---

## Evaluation

```bash
python3 test.py --name WebOfScience-run1 [--extra {_macro,_micro}] [--batch 32]
```

Reports: **Macro-F1**, **Micro-F1**, **P@1**, **P@3**, **P@5**, **R-Precision**.

R-Precision requires at least one gold label per sample. Evaluation fails
explicitly if that data invariant is violated.

## Five-seed workflow

From this directory:

```bash
bash scripts/run_seed_sweep_sequential.sh
```

`GPU_ID`, `TRAIN_BATCH`, `EVAL_BATCH`, `SEEDS`, `DATASETS`, `CONDA_ENV_NAME`
and `PYTHON_BIN` can be set through environment variables. The aggregation
scripts read `checkpoints/{DATASET}-seed{SEED}/cost_metrics.json`.

---

## Output Files

Both files are written to `checkpoints/{DATA-NAME}/`:

**`log.txt`** — per-epoch Macro-F1 and Micro-F1.

**`cost_metrics.json`** — populated by `train.py` and extended by `test.py`.
It records run identity, model parameter counts, training timing, inference
timing and the evaluated test metrics. Treat it as a local audit artifact; do
not commit run-specific copies to Git.

The historical server environment was used during release intake but is not
published as a lock file, because it contains machine-specific audit detail and
is not a portable installation promise. Use the reconstruction commands above
as the release-facing setup record.

# XR-Transformer environment guide

This guide describes the historical study environment as far as it can be
reconstructed. XR-Transformer and PECOS are third-party software from
[`amzn/pecos`](https://github.com/amzn/pecos). The study integration and its
verification limits are documented in
[`XMLmodels/pecos/STUDY_INTEGRATION.md`](XMLmodels/pecos/STUDY_INTEGRATION.md).

## Historical constraints

The student guide used Python 3.9. The copied PECOS `setup.py` requires:

- NumPy 1.19.5 or newer;
- SciPy 1.4.1 or newer;
- scikit-learn 0.24.1 or newer;
- PyTorch 1.8 or newer, but older than 2.0;
- SentencePiece 0.1.86 or newer, excluding 0.1.92;
- Transformers 4.4.2 or newer on Python 3.9.

PECOS also builds a C++17/OpenMP extension and discovers BLAS/LAPACK through
NumPy. A Linux build therefore needs a compiler toolchain plus BLAS/LAPACK
development libraries. The exact historical CUDA, compiler and transitive
package versions were not preserved.

## Installation attempt

Create an isolated environment and install the copied PECOS tree in editable
mode:

```bash
conda create --name xr-transformer-release python=3.9 pip -y
conda activate xr-transformer-release

# Install a suitable compiler, BLAS/LAPACK and CUDA-enabled PyTorch for the host.
cd XMLmodels/pecos
python -m pip install --editable .
```

The editable installation compiles `pecos.core.libpecos_float32`. A successful
install should be captured with:

```bash
python --version
python -m pip freeze
python -c "import pecos; from pecos.utils.smat_util import Metrics; print('PECOS import OK')"
```


## Data preparation

Prepare HTC-format text and comma-separated zero-based label files using:

```bash
python XMLPreprocessing/XR-Transformer/preprocess.py \
  --input-dir /path/to/input \
  --output-dir XMLmodels/pecos/htc-base/wos
```

The output contains `X.trn.txt`, `X.tst.txt`, `Y.trn.npz`, `Y.tst.npz` and the
TF-IDF matrices under `tfidf-attnxml/`. Dataset files are intentionally ignored
by Git.

Original XML benchmark datasets were historically downloaded from the
[PECOS dataset archive](https://archive.org/download/pecos-dataset/xmc-base/).
Users are responsible for the terms of each source dataset.

## Study launchers

Run from the ensemble directory so its relative paths resolve:

```bash
cd XMLmodels/pecos/run_ensemble
bash run.sh wos htc-base
```

The second argument names a dataset-root directory located one level above
`run_ensemble`; the example resolves to `XMLmodels/pecos/htc-base/wos`.
Training refuses to overwrite an existing `models/wos` directory. To evaluate
already generated model predictions:

```bash
bash run_eval_only.sh wos htc-base
```

Both commands use the historical parameter files under `params/`. Their syntax
was checked during release preparation, but the configurations and retained
compact results still require reconciliation with the final paper tables.

# HBGL study integration

This directory contains the HBGL implementation adapted for the cross-domain
HTC/XMC evaluation. HBGL was developed by Ting Jiang, Deqing Wang, Leilei Sun,
Zhongzhi Chen, Fuzhen Zhuang and Qinghong Yang; it is not a model introduced by
this project.

- Original repository: <https://github.com/kongds/HBGL>
- Inspected base commit: `a40acdf87407a5a6cdd4c921c80c60b9f3522aa1`
- Original paper: [*Exploiting Global and Local Hierarchies for Hierarchical
  Text Classification*](https://arxiv.org/abs/2205.02613)
- Verification status: **Documented** - metric helpers, Python syntax and shell
  syntax are tested, but no fresh GPU run has been completed from this checkout.

See [the modification record](../../docs/HBGL_MODIFICATIONS.md) for the bounded
upstream comparison, fixes and known limitations.

## Setup

The preserved student environment specifies Python 3, PyTorch 2.5.1 and
Transformers 4.17.0. Upstream HBGL instead pinned Transformers 2.10.0 and did
not pin PyTorch. This substantial version difference is part of the study fork,
not an upstream recommendation. The student combination has not yet been
reproduced in a clean environment:

```bash
cd htc/hbgl
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the CUDA-specific PyTorch build appropriate for the target machine if
the default package is unsuitable. After a successful installation, record
`python --version`, `python -m pip freeze`, `nvidia-smi` and the exact Git
commit rather than treating the current requirements as a universal lockfile.

## Data and preprocessing

Each dataset directory under `data/` must contain:

```text
<dataset>_train.json
<dataset>_val.json
<dataset>_test.json
<dataset>.taxonomy
```

Input JSON Lines records use `{"token": <text>, "label": [<label>, ...]}`.
The generic preprocessor writes the HBGL training/evaluation files and label map:

```bash
cd scripts
python ../preprocess.py wos 3
```

Raw and generated dataset JSON, caches, checkpoints, model files, results and
logs are excluded from Git. Dataset acquisition and taxonomy redistribution
must be handled according to each source dataset's terms.

## Study commands

Run the preserved five-seed launchers from `htc/hbgl/scripts`; they use seeds
`42, 1, 2, 3, 4`, keep each seed in a separate output directory, and configure
Weights & Biases in offline mode:

```bash
cd htc/hbgl/scripts
bash run_wos.sh
bash run_nyt.sh
bash run_rcv1.sh
bash run_ac13.sh
bash run_wiki10-31.sh
```

`run_a670-OOM.sh` is retained as an explicitly unverified Amazon-670K attempt,
not as a working paper command. Existing output directories are never deleted by
these launchers; choose a new run name or archive an earlier run first.

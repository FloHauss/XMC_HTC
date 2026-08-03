# HBGL study integration

This directory contains the HBGL implementation adapted for the cross-domain
HTC/XMC evaluation. HBGL was developed by Ting Jiang, Deqing Wang, Leilei Sun,
Zhongzhi Chen, Fuzhen Zhuang and Qinghong Yang; it is not a model introduced by
this project.

- Original repository: <https://github.com/kongds/HBGL>
- Inspected base commit: `a40acdf87407a5a6cdd4c921c80c60b9f3522aa1`
- Original paper: [*Exploiting Global and Local Hierarchies for Hierarchical
  Text Classification*](https://arxiv.org/abs/2205.02613)
- Repository note: the adapted source, preprocessing and study launchers are
  included. The preserved dependency combination should be treated as a study
  environment record rather than a universal setup.

## Upstream and provenance

The local source was added in commit `62ad1a3` and compared with
`kongds/HBGL` commit `a40acdf87407a5a6cdd4c921c80c60b9f3522aa1`. It retains
substantial HBGL and Microsoft UniLM `s2s-ft` source. The original student fork
history was not preserved; line-ending conversion accounts for much of the
apparent `s2s_ft` diff, while the study changes are bounded in the remaining
files. No clear top-level licence was found in the inspected upstream checkout,
so permission or licence clarification is still needed before redistribution is
settled.

## Study-specific changes

The fork refactors upstream training into `main/`, adds offline W&B logging,
tester helpers, newer Transformers compatibility, label-initialisation options,
split conceptual pre-training, multiprocessing/cached feature creation,
expanded decoding metrics and cross-domain configurations. The working paper
launchers preserve BERT base uncased, five seeds (`42, 1, 2, 3, 4`) and the
following settings: learning rate `3e-5`, 96,000 training steps and 500 warm-up
steps. The Amazon-670K launcher is explicitly an OOM, unverified attempt.

| Dataset | Source / target length | Batch | Study-specific configuration |
| --- | ---: | ---: | --- |
| WOS | 509 / 3 | 12 | Conceptual pre-training, BCE, 300 steps |
| NYT | 472 / 9 | 12 | Leaf-name initialisation, conceptual pre-training, BCE, 1,000 steps at `1e-4` |
| RCV1 | 492 / 5 | 12 | Expanded label descriptions, random initialisation, conceptual pre-training, BCE, 100 steps |
| AmazonCat-13K | 500 / 4 | 12 | Random initialisation; ignore synthetic meta-labels |
| Wiki10-31K | 500 / 4 | 12 | Random initialisation; ignore synthetic meta-labels |
| Amazon-670K | 500 / 5 | 16 | Random initialisation; known OOM attempt |

Correctness fixes reject malformed/empty gold labels, safely handle short
predictions, honour `--no_cuda`, repair CPU and TensorBoard paths, prevent
duplicate checkpoint decoding and cross-dataset embedding reuse, and preserve
separate seed output directories without deleting earlier runs.

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

## Verification status and limitations

No training or inference path has been run in a fresh environment, and multi-GPU
behaviour has not been revalidated after the refactor. The preserved requirements
are a historical study record, not a lockfile. Conceptual pre-training and the
`main/` refactor lack model-level tests; multiprocessing portability and peak
memory are unmeasured. P@k deliberately uses denominator `k` for short decoded
vectors, matching standard XMC precision semantics. Retained taxonomy and
label-vocabulary files may derive from source datasets and require a separate
redistribution review.

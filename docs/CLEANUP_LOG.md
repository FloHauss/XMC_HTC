# Release-preparation cleanup log

This log records material removed from the working release tree. All removals
remain recoverable from Git history until any separately approved history
rewrite occurs.

## 2026-07-17

Removed 60 exact tracked paths:

- 57 raw Slurm/debug `.out` files and notebook-checkpoint duplicates under
  `XMLmodels/pecos/run_ensemble/results`;
- `XMLmodels/pecos/pecos/core/libpecos_float32.cpython-39-x86_64-linux-gnu.so`,
  a platform-specific compiled Python 3.9 extension;
- `htc/hbgl/data/rcv1/preprocess/lyrl2004_tokens_train.dat`, approximately
  17.8 MB of derived RCV1 token data;
- `.gitmodules`, whose two PECOS declarations no longer represented active Git
  submodules.

The compact XR-Transformer `average.txt` summaries, aggregation/launch scripts,
source modifications and hierarchy metadata were retained. No broader PECOS
source reduction was attempted in this step.

Removed three derived NYT JSON files under `htc/hbgl/data/nyt/preprocess/` that
contained tokenised text examples. The generic preprocessor, taxonomy and label
vocabulary were retained. Dataset JSON beneath HBGL preprocessing directories is
now ignored to prevent accidental recommits. The removed files remain
recoverable from Git history.

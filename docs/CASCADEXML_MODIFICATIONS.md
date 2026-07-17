# CascadeXML modification record

## Scope and upstream base

The local source was added to this repository in commit `68e2c77` on
2026-02-10. A file-by-file comparison was made against
`xmc-aalto/cascadexml` commit
`ce701f688aaf5d5c8abe979d192f9c8f224aec90` (2024-02-23).

Seven retained files are byte-identical to that upstream commit:
`Runner_sparse.py`, `dist_eval_sampler.py`, `log.py`, `random_walks.py`,
`training_schedule.py`, and `tree.py`; upstream `main_inference.py` was also
identical before release-readiness fixes. The shared unchanged files and bounded
diff support using `ce701f6` as the inspected base. No earlier local commit or
patch records the students' original editing process, so this is provenance by
comparison rather than a preserved fork history.

The original upstream checkout has no top-level licence file. Attribution is
recorded, but licence or permission still needs to be settled before public
release.

## Study-specific changes

| File | Result-affecting changes |
| --- | --- |
| `CascadeXML.py` | Adds DeBERTa base/large and BERT-large encoders; adds transformer-layer schedules for two- and five-level cascades and corresponding large encoders. |
| `Runner.py` | Adds configurable P@k through 20, R-Precision, thresholded micro/macro/sample F1 and exact-match accuracy; selects early stopping by micro-F1; records training time; saves prediction matrices. |
| `data_utils.py` | Adds the new tokenisers and inverse-propensity constants for cross-domain datasets. |
| `dataset.py` | Adds the new tokenisers and an Amazon-3M-specific memory shortcut that retains 20 label features before graph propagation. |
| `main.py` | Adds cross-domain dataset names, `--max_patience`, configurable evaluation `--top_k`, and different tree-depth and graph defaults. |
| `main_inference.py` | Release fixes make the historical entry point instantiate the current model and accept custom label counts and output depth. |
| `io_utils.py` | Release-specific, dependency-light label and TF-IDF conversion with input validation. |

`Runner_old.py` was removed during release preparation. It was unreferenced and
differed from upstream `Runner.py` only by P@2 display text and whitespace; the
file remains recoverable from Git history.

## Release-readiness fixes

- Replaced process-terminating early stopping with a cooperative return and DDP
  stop broadcast.
- Fixed DDP access to the model patience setting and made metric buffers use the
  selected device.
- Restored upstream best-checkpoint saving, which had been commented out.
- Saves unwrapped model weights under DDP and accepts historical DDP-prefixed
  weights in the standalone inference entry point.
- Rejects evaluation cutoffs below the metrics' required P@5 and rejects empty
  gold sets for R-Precision.
- Fixed LF parsing that previously consumed the input before iteration and
  corrected the `read_lf_datasets` call typo.
- Fixed the stale inference constructor, tokenizer selection, label-map handling,
  parser choices and support for datasets outside the upstream hard-coded map.
- Removed unused experimental batching code and debug output without changing
  the Amazon-3M shortcut.

## Known scientific and operational limitations

- The source compiles and conversion tests pass, but training and inference have
  not been smoke-tested with PyTorch, Transformers, GPU hardware, model weights,
  or a complete dataset in a fresh environment.
- The exact Python/CUDA/dependency environment and the commands used for paper
  results were not committed. The upstream installer is unpinned and modifies a
  `pyxclib` checkout, so it is not yet a reproducible environment specification.
- The source contains no paper configuration files. Parser defaults, including
  `tree_depth=[9, 12]`, `top_k=20`, and `num_labels=670091`, must not be presented
  as settings for every dataset.
- The chosen inverse-propensity `A` and `B` constants for NYTimes, WOS, RCV1,
  Ohsumed, 20ng, GoEmotions and EconBiz lack an in-repository rationale. They
  must be reconciled with the analysis protocol before propensity-scored results
  are promoted.
- The Amazon-3M pre-propagation top-20 shortcut changes graph features relative
  to upstream and has no recorded validation.
- Thresholded F1 is computed only over the returned top-k shortlist. It is not a
  dense-label F1 calculation and should be described accordingly in the paper.
- With `--dist_eval`, P@k counts are reduced across ranks, but the F1 input lists
  are not gathered. F1 and F1-based early stopping are therefore valid only for
  single-process evaluation (`--dist_eval` disabled) until that path is fixed and
  tested.

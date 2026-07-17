# XR-Transformer / PECOS modifications

This record distinguishes study adaptations and release-preparation fixes from
the third-party PECOS/XR-Transformer implementation.

## Comparison base

- Upstream repository: <https://github.com/amzn/pecos>
- Historical submodule revision:
  `3fccd9af1b287c1cab96a7a16e93c2ff0bfbc903`
- Copied-tree conversion commit:
  `1606f625070fb1ea8977cc1aa610ffa9e6b2bada`
- Licence: Apache-2.0.

The exact former submodule revision is preserved in Git history. The conversion
commit copied the full checkout into this repository and introduced study
material. It does not preserve enough information to distinguish any
uncommitted submodule edits from files added during conversion. Subsequent
parent-repository commits provide a bounded record of later study changes.

The retained directory currently contains about 451 tracked files and 11 MB of
material, including PECOS components unrelated to the paper workflow. A broader
reduction to a pinned dependency plus a small study overlay is deferred because
it would carry more regression risk than the current publication clean-up.

## Historical study adaptations

The parent repository history and current integration show the following study
work:

- parameter sets and launch scripts for HTC and XMC datasets;
- BERT, RoBERTa and XLNet training/prediction variants and their ensembles;
- evaluation output extended from the upstream top-10 display to top 20;
- R-Precision and F1 reporting added to the evaluation workflow;
- cross-domain data conversion under `XMLPreprocessing/XR-Transformer`;
- compact per-seed and aggregate result text files retained after removal of
  raw scheduler logs.

## Release-preparation fixes

The release-preparation branch makes these bounded corrections:

- the converter now has an explicit command-line interface, validates row and
  label invariants, produces conventional output names and gives train/test
  matrices a shared label dimension;
- R-Precision rejects empty-gold samples and handles prediction rows shorter
  than the requested rank without crashing;
- ensemble F1 is computed directly from sparse supports, avoiding dense
  materialisation of extreme-label matrices;
- the requested PECOS ensemble method is now used consistently for both ranked
  and F1 evaluation;
- the upstream example evaluator is restored to valid Python after historical
  argument edits left syntax errors and a stale propensity parameter;
- launch scripts use strict shell error handling and quoted arguments, propagate
  failures through `tee`, and no longer recursively delete an existing model
  directory;
- evaluation-only scripts accept a dataset argument instead of hard-coding WOS;
- a dead commented experimental gradient block was removed from `matcher.py`.

## Result-affecting caveats

- F1 is calculated over every stored non-zero prediction. It is a sparse
  support metric, not dense thresholded multilabel F1.
- R-Precision values produced before the empty-gold validation fix have not been
  audited against the original generated matrices.
- Historical `average.txt` files for Amazon-670K, AmazonCat-13K and Wiki10-31K
  contain five identical per-seed records and zero standard deviations. Several
  HTC datasets contain four identical records out of five. This may reflect
  copied or reused outputs rather than independent runs.
- The retained compact results are therefore historical and unreconciled. They
  are not evidence of a new release verification run and must not be promoted
  to final paper tables without checking the archived experiment records.
- No fresh dependency installation or representative GPU training run has been
  completed for the release tree.

## Automated evidence

`tests/test_xr_transformer.py` covers:

- short sparse prediction rows and empty-gold R-Precision policy;
- sparse support-F1 calculations;
- shared train/test label dimensions in preprocessing;
- rejection of empty gold-label rows.

Continuous integration also parses/compiles the release-facing Python and shell
entry points. These checks support a **Documented** status, not a claim of
end-to-end reproduction.

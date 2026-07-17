# XR-Transformer study integration

This directory contains the PECOS implementation of XR-Transformer together
with adaptations made to evaluate the model in the cross-domain HTC/XMC study.
XR-Transformer and PECOS were developed by their original authors; they are not
models introduced by this project.

## Upstream and licence

- Original repository: <https://github.com/amzn/pecos>
- Former submodule revision:
  `3fccd9af1b287c1cab96a7a16e93c2ff0bfbc903`
- Licence: Apache-2.0; see [`LICENSE`](LICENSE) and the upstream notice files in
  this directory.

Git history shows that the repository originally pinned PECOS as a submodule at
the revision above. Commit `1606f625070fb1ea8977cc1aa610ffa9e6b2bada`
converted that submodule into a copied source tree. The conversion preserved a
complete PECOS checkout and added study files, but cannot prove whether the
submodule working tree also contained uncommitted edits at that moment. Changes
made after the conversion are bounded by the parent repository history.

The detailed comparison and release fixes are recorded in
[`docs/XR_TRANSFORMER_MODIFICATIONS.md`](../../docs/XR_TRANSFORMER_MODIFICATIONS.md).

## Study-specific entry points

- [`run_ensemble/`](run_ensemble/) contains the historical multi-encoder
  training, prediction and ensemble evaluation workflow.
- [`../../XMLPreprocessing/XR-Transformer/preprocess.py`](../../XMLPreprocessing/XR-Transformer/preprocess.py)
  converts the study's text and label files to the names and sparse matrices
  expected by XR-Transformer.
- [`run_ensemble/params/`](run_ensemble/params/) contains historical parameter
  files for the cross-domain datasets and encoder variants.

The preprocessing command is:

```bash
python XMLPreprocessing/XR-Transformer/preprocess.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output
```

The input directory must contain `train_raw_texts.txt`,
`test_raw_texts.txt`, `train_labels.txt` and `test_labels.txt`. Each label line
is a comma-separated set of zero-based integer label identifiers. The converter
rejects empty gold-label rows and uses one shared label dimension for the train
and test matrices.

The ensemble launch scripts provide their own `--help` or usage output. They
expect a working PECOS/XR-Transformer environment and the preprocessed dataset
layout used by the study workflow. See
[`xr_transformer_guide.md`](../../xr_transformer_guide.md) for the reconstructed
environment constraints.

## Evaluation semantics

The integration reports P@k, recall@k and R-Precision using PECOS sparse ranked
predictions. R-Precision is undefined for samples without gold labels, so the
release evaluator rejects such data.

The ensemble evaluator also reports micro- and macro-F1 over the stored sparse
prediction support. This is not equivalent to thresholding a dense score matrix
at a paper-selected decision threshold. These F1 values must therefore be
described as support F1 unless a separate threshold protocol is specified.

## Repository note

The provenance, parameter syntax, preprocessing invariants, metric edge cases
and shell syntax are covered by repository checks. The retained compact XR
outputs are archival and mostly do not correspond to the final paper tables;
use [`RESULTS.md`](../../RESULTS.md) for the reported results.

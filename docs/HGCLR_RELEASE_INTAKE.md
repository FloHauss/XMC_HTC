# HGCLR release intake

This document reconciles the 2026-07-17 server-side audit of the working HGCLR
fork with the XMC/HTC release-readiness audit.

## Source state

- Working checkout: server-side `contrastive-htc` checkout
- Working branch: `main`
- Upstream: <https://github.com/wzh9969/contrastive-htc>
- Upstream reference: `322a7ff2d83c878534bed25bb288cf4479d00363`
- Licence: MIT
- Conda environment used historically: `contrastive-htc`
- Server audit states that all release changes remain uncommitted.

The canonical method abbreviation used in release-facing material is **HGCLR**.

## Candidate release content

### Modified upstream files

| File | Study-specific purpose | Required review |
| --- | --- | --- |
| `eval.py` | Adds P@1, P@3, P@5 and R-Precision | Unit-test definitions and check the empty-gold denominator. |
| `utils.py` | Adds wall-clock, throughput, GPU-memory and parameter cost tracking | Check phase boundaries, peak reset behaviour and JSON merge semantics. |
| `train.py` | Records cost metrics and restores legacy NumPy aliases for fairseq 0.10 | Replace the `eval()`-based compatibility patch with explicit aliases and a rationale. |
| `test.py` | Records test/cost metrics and loads legacy checkpoints under modern PyTorch | Explain the trusted-checkpoint requirement for `weights_only=False`. |

### Untracked source and documentation

- `USAGE.md`
- `data/{WebOfScience,nyt,rcv1}/preprocess_from_htc.py`
- `data/binarize.py`
- `run_seed_sweep_sequential.sh`
- `aggregate_seed_results.py`
- `aggregate_webofscience_results.py`
- `summarize_seed_sweep.py`
- bwUniCluster setup, submission and job scripts, after portability review
- final `*_seed_aggregate.csv` and `*_seed_aggregate.json` files, now reconciled
  with the displayed final paper results

These files should not be described as a new HGCLR implementation. They are the
integration and experimental support needed to apply the original model in this
study.

## Excluded generated content

- `checkpoints/` - approximately 14 GB
- binarised `data/*/{Y,tok}.{bin,idx}`
- `slot.pt`, `split.pt` and `bert_value_dict.pt`
- caches, raw scheduler logs and environment-local state

Trained checkpoints may be archived separately if there is a concrete reuse
case, but they are not required in Git. Generated binarised datasets must be
recreated from documented preprocessing.

## Scientific checks before integration

### R-Precision

The audited `_r_precision` implementation skips calculation for samples with no
gold labels but divides the accumulated score by the total number of predictions.
This is equivalent to assigning zero to empty-gold samples, not excluding them
from the mean.

Before changing it:

1. establish the intended definition used consistently across all models;
2. count empty-gold examples in every evaluated split after preprocessing;
3. determine whether empty-gold samples occurred in the historical binaries;
4. add unit tests for ordinary, variable-cardinality and empty-gold cases;
5. if results change, regenerate affected aggregates or document why they do not.

### Compatibility patches

The NumPy alias compatibility code is load-bearing for `fairseq==0.10.0` under
NumPy 1.24 or newer, but the reported `eval()` fallback is unnecessarily opaque.
The release version should use an explicit mapping and explain that it is a
legacy dependency shim.

`torch.load(..., weights_only=False)` is needed for trusted historical
checkpoints after the PyTorch 2.6 default change. The release must state that
untrusted checkpoint files must not be loaded this way.

## Verification target

The source-only intake is now present and the integration status is
**Documented**. It may be raised to **Smoke-tested** after:

- a clean environment can import the model and preprocessing code;
- a tiny or bounded dataset can be binarised;
- one training/evaluation path completes;
- metric and aggregation tests pass;
- no machine-specific path is required.

A full five-seed rerun is not required merely to label the integration
Smoke-tested. Historical aggregates should remain explicitly distinguished from
new verification evidence.

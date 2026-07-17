# Reproducibility guide

This repository combines third-party research implementations with mutually
incompatible historical dependency stacks. There is intentionally no single
training environment for all models. The small root environment only validates
study-owned conversion, evaluation and release-facing files.

## Repository-level checks

With Python 3.10 or later:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-release.txt
bash scripts/check_release.sh
```

These checks require no datasets, model downloads or GPU. Passing them confirms
metric/conversion invariants, documentation links, Python syntax, shell syntax
and XR-Transformer JSON parameter syntax. It is not an end-to-end model test.

## Integration environment matrix

| Integration | Environment evidence | Current verification | Setup record |
| --- | --- | --- | --- |
| CascadeXML | Upstream package list, reconstructed in this repository | Dependencies not freshly installed; no GPU run | [`XMLmodels/CascadeXML/README.md`](../XMLmodels/CascadeXML/README.md) |
| XR-Transformer | Historical PECOS `setup.py` constraints and student Python 3.9 guide | Source-level checks only; compiled extension and GPU run unverified | [`xr_transformer_guide.md`](../xr_transformer_guide.md) |
| HBGL | Student requirements preserved; upstream used a substantially older Transformers version | Dependencies not freshly installed; no GPU run | [`htc/hbgl/README.md`](../htc/hbgl/README.md) |
| HGCLR | Complete server environment export plus a shorter reconstruction command | Historical server environment captured; release checkout not freshly GPU-tested | [`integrations/hgclr/USAGE.md`](../integrations/hgclr/USAGE.md) |
| RADAr | Study implementation unavailable | Deferred | Upstream link only in [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md) |

## Verification boundaries

For publication, record evidence separately for each integration. Do not raise
an integration above **Documented** merely because the repository-level checks
pass. A representative model smoke test should record:

1. operating system, Python, CUDA, GPU and dependency versions;
2. exact Git commit and whether the working tree was clean;
3. dataset name and a checksum or immutable source reference;
4. preprocessing and launch commands;
5. exit status, runtime and the location of compact metrics;
6. any deviation from the historical paper configuration.

Large full-paper reruns are not required to establish a bounded smoke test. A
small representative dataset or deliberately limited training schedule is
acceptable if it exercises preprocessing, model construction, one optimisation
step, checkpoint writing, loading and evaluation, and is labelled accordingly.

## Historical results

Existing result files are provenance records rather than fresh verification.
HGCLR candidate aggregates and XR-Transformer compact averages remain
unreconciled with the final paper tables. XR-Transformer files with repeated
per-seed records must be checked against the experiment archive before use.

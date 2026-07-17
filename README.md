# Cross-domain evaluation of XMC and HTC models

This repository accompanies a study of how established hierarchical text
classification (HTC) and extreme multi-label classification (XMC) models behave
when applied to datasets from the other domain.

The evaluated models were developed by their respective original authors. This
project does **not** claim ownership of CascadeXML, XR-Transformer, HBGL, HGCLR
or RADAr. Its contribution is the cross-domain integration: dataset conversion,
hierarchy preparation, configurations, evaluation, cost measurement and
empirical comparison.

> **Release status:** preparation in progress. Historical code and results are
> being consolidated into an honest research artefact. A component marked
> **Documented** is present but has not necessarily completed a new end-to-end
> run from a fresh checkout.

## Evaluated methods

| Method | Original implementation | Repository integration | Status |
| --- | --- | --- | --- |
| CascadeXML | [xmc-aalto/cascadexml](https://github.com/xmc-aalto/cascadexml) | [`XMLmodels/CascadeXML`](XMLmodels/CascadeXML) | Documented |
| XR-Transformer | [amzn/pecos](https://github.com/amzn/pecos) | [`XMLmodels/pecos`](XMLmodels/pecos/STUDY_INTEGRATION.md) | Documented |
| HBGL | [kongds/HBGL](https://github.com/kongds/HBGL) | [`htc/hbgl`](htc/hbgl) | Documented |
| HGCLR | [wzh9969/contrastive-htc](https://github.com/wzh9969/contrastive-htc) | [`integrations/hgclr`](integrations/hgclr) | Documented |
| RADAr | [yousef-younes/RADAr](https://github.com/yousef-younes/RADAr) | Not currently available | Deferred |

The status definitions and evidence requirements are recorded in
[`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md). Model provenance is
tracked in [`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md).

## What this repository provides

- conversion between the HTC JSONL and XMC text/label representations;
- hierarchy and taxonomy preparation used for cross-domain datasets;
- adapted launch and preprocessing scripts for the evaluated methods;
- Micro-F1, Macro-F1, P@k and R-Precision evaluation support where applicable;
- multi-seed aggregation and computational-cost records;
- historical configurations and candidate result records;
- explicit provenance, verification status and known limitations.

## Repository map

```text
dataset_transfer/          HTC/XMC format conversion
XMLPreprocessing/          XML preprocessing helpers
XMLScripts/                XML hierarchy and feature scripts
XMLmodels/CascadeXML/      Adapted CascadeXML source
XMLmodels/pecos/           Historical XR-Transformer/PECOS working tree
htc/hbgl/                  Adapted HBGL source
integrations/hgclr/        Audited HGCLR study integration
tests/                     Bounded release-validation tests
docs/                      Provenance, inventory and release-readiness records
```

The copied PECOS tree and the older model directories are retained temporarily
while their study-specific modifications are isolated. Their current layout
should not be interpreted as the intended final release structure.

## Getting started

There is currently no single environment that supports every evaluated model.
Several implementations depend on different legacy versions of PyTorch,
Transformers, fairseq or PECOS. Follow the documentation for the integration you
want to use:

- [HGCLR integration and status](integrations/hgclr/README.md)
- [HGCLR usage](integrations/hgclr/USAGE.md)
- [HTC/XMC dataset conversion](dataset_transfer/README.md)
- [XR-Transformer integration and status](XMLmodels/pecos/STUDY_INTEGRATION.md)
- [historical XR-Transformer guide](xr_transformer_guide.md)
- [historical model and preprocessing instructions](docs/LEGACY_USAGE.md)

The legacy instructions are preserved for provenance and still require
verification and editing. Paths or commands appearing only there should not yet
be treated as release-tested interfaces.

## Data

Datasets are not distributed as a complete part of this repository. NYT and
RCV1-V2 in particular have acquisition or redistribution conditions that users
must satisfy independently. Dataset-specific instructions should be used to
obtain the original data and reproduce the derived representations.

Generated model inputs (`.bin`, `.idx`, `.pt`), checkpoints, caches and raw
scheduler logs are intentionally excluded from Git. Taxonomies or label metadata
retained in the release must be accompanied by their provenance and generation
procedure.

## Results and reproducibility

Historical outputs are being reduced to compact, machine-readable result files.
Raw Slurm logs in the current tree are not the intended release interface.

HGCLR candidate five-seed aggregates are retained under
[`integrations/hgclr/results/candidate`](integrations/hgclr/results/candidate).
They are historical records and have not yet been reconciled with the final
paper tables. They must not be interpreted as new verification runs.

The current repository audit and proposed disposition of historical material are
documented in [`docs/REPOSITORY_INVENTORY.md`](docs/REPOSITORY_INVENTORY.md).

## Validation

The bounded release checks cover metric and conversion invariants for HGCLR,
CascadeXML, HBGL and XR-Transformer, as well as release-document links:

```bash
python -m unittest discover -s tests -v
```

Full GPU training is intentionally not part of the local test suite or future
continuous integration.

## Attribution

Please cite the original paper for every model used. Per-integration provenance
and upstream licences or permissions will be retained with the corresponding
code; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). A citation for the
cross-domain study will be added when the publication metadata is final.

Substantial implementation and experiment work in this repository was conducted
as part of a student group project. The Git history is preserved so those
contributions remain attributable.

## Known limitations

- RADAr study code is currently unavailable and its integration is deferred.
- CascadeXML, XR-Transformer and HBGL have not yet completed fresh-checkout
  release verification.
- The historical PECOS tree contains considerably more upstream material than
  the study needs.
- Some legacy scripts and raw outputs still require removal or consolidation.
- The release does not aim to provide a unified production API for the five
  third-party models.

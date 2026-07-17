# Repository inventory and proposed disposition

This inventory records material found at the audited `main` commit
`d011395f43e44177b0e82541edf98522e088c634`. Proposed dispositions are not
deletions: historical outputs and working copies must be backed up before the
public tree is reduced.

## Top-level areas

| Path | Approximate checkout size | Current purpose | Proposed disposition |
| --- | ---: | --- | --- |
| `XMLmodels/pecos` | 23 MB | Copied PECOS/XR-Transformer source, study scripts and raw results | Separate the pinned upstream dependency from study-owned scripts and compact results. |
| `htc/hbgl` | 20 MB | Modified HBGL source, preprocessing material and run scripts | Preserve the integration, remove corpus/generated artefacts, and document its upstream diff. |
| `XMLmodels/CascadeXML` | 188 KB | Modified CascadeXML source | Preserve initially; establish provenance and fix confirmed entry-point blockers. |
| `XMLPreprocessing` | 24 KB | XML model preprocessing | Consolidate only after checking duplication and paper usage. |
| `XMLScripts` | 24 KB | Additional XML preprocessing/clustering scripts | Determine whether these are superseded duplicates before retaining or removing them. |
| `dataset_transfer` | 32 KB | Study-specific HTC/XMC conversions | Treat as a core contribution; test and document it. |

## Generated and platform-specific material

The following findings describe the audited baseline. The raw study Slurm
outputs, their notebook duplicates and the compiled PECOS extension have since
been removed from the release-preparation tree; see
[`CLEANUP_LOG.md`](CLEANUP_LOG.md).

- 74 tracked `.out`, `.log`, notebook-checkpoint or compiled `.so` files were
  detected.
- `XMLmodels/pecos/run_ensemble/results` contains 63 files and occupies about
  11 MB. Most are raw Slurm logs rather than compact result records.
- The Slurm logs expose historical cluster paths and student account names.
- `XMLmodels/pecos/pecos/core/libpecos_float32.cpython-39-x86_64-linux-gnu.so`
  is a platform-specific compiled extension and should not be distributed as
  source.
- The copied PECOS tree includes unrelated examples, images, tutorials and its
  own repository administration files.

Proposed action: preserve raw logs in a private archival bundle if they are
still needed for provenance, extract the paper-relevant metrics and run metadata
into compact machine-readable files, and remove raw scheduler output and the
compiled extension from the public tree.

## Dataset and hierarchy material

The HBGL tree contains preprocessing scripts, hierarchy metadata and derived
corpus material. These categories must not be treated identically.

### Requires removal or explicit redistribution review

- `htc/hbgl/data/rcv1/preprocess/lyrl2004_tokens_train.dat` - approximately
  17.8 MB of RCV1 token data. Removed from the release-preparation tree.
- `htc/hbgl/data/nyt/preprocess/idnewnyt_{train,val,test}.json` - derived lists
  from the licensed NYT corpus. Removed from the release-preparation tree; they
  remain recoverable from Git history.

### Likely release metadata, subject to provenance checks

- taxonomy files;
- label vocabularies and maps;
- preprocessing scripts;
- synthetic clustering-derived hierarchies for XML datasets.

Proposed action: document how every released hierarchy was generated and ensure
that it contains no underlying document text. Dataset download and preparation
instructions should replace redistributed corpus content.

## Dependency and provenance inconsistencies

- At the audited baseline, `.gitmodules` declared both `submodules` and
  `XMLmodels/pecos`, but the repository had no active submodule entries. The
  stale file has since been removed.
- Git history shows that `XMLmodels/pecos` was converted from a pinned submodule
  into a normal directory in 2024.
- At the audited baseline CascadeXML had an empty `requirements.txt`. A
  clearly labelled, upstream-derived package inventory now replaces it; exact
  compatible versions remain unverified.
- At the audited baseline the repository had no shared environment or
  validation entry point. A bounded root requirements file and release-check
  command now cover study-owned utilities without imposing one training stack.
- The five integrations may require different legacy environments; a single
  forced environment is not assumed to be realistic.

The small common environment is now recorded in `requirements-release.txt`.
Model-specific records preserve known constraints and clearly identify the
remaining unverified dependency stacks.

## Confirmed code and documentation blockers

- At the audited baseline, HGCLR and RADAr were absent. HGCLR has since been
  imported under `integrations/hgclr`; RADAr is deferred because the study copy
  is unavailable.
- CascadeXML `main_inference.py` imports a missing `Runner_accelerate` module.
- The root README points to the absent `XMLmodels/CascadeXML/src` directory.
- The root README does not clearly distinguish third-party models from the
  study's contributions.
- No automated test or CI configuration exists at the repository root.

## Safe order of operations

1. Preserve server implementations and raw experiment records.
2. Extract provenance and paper-relevant configurations.
3. Decide the retained public representation of each upstream model.
4. Add tests around conversion and metrics.
5. Remove generated, restricted and redundant material.
6. Rewrite the paper-facing documentation against the resulting tree.
7. Verify from a fresh checkout and record an honest status for each model.

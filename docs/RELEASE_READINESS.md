# Release readiness

This document tracks preparation of the repository accompanying the cross-domain
HTC/XMC study. The evaluated models are third-party research implementations.
The repository's contribution is their cross-domain integration, including data
conversion, hierarchy handling, experiment configurations, evaluation and
results.

The objective is a solid and transparent research artefact. It is not to turn
the evaluated implementations into a uniform production library or to imply
that every historical experiment has been rerun during release preparation.

## Audited baseline

- Repository: `FloHauss/XMC_HTC`
- Default branch: `main`
- Audited commit: `d011395f43e44177b0e82541edf98522e088c634`
- Audit date: 2026-07-17
- Release-preparation branch: `release-preparation` (pushed for review)

## Verification vocabulary

Every integration must use one of the following statuses. A status must only be
raised when the corresponding evidence has been recorded.

| Status | Meaning |
| --- | --- |
| Verified | Installed from a fresh checkout and completed a representative end-to-end run. |
| Smoke-tested | Imports, preprocessing, configuration and a bounded execution path were tested. |
| Documented | Code and historical instructions are present, but current execution has not been independently verified. |
| Incomplete | A required component, configuration or instruction is known to be missing. |

## Release floor

The publication release should meet the following minimum standard:

- [ ] All currently available model integrations are present or reproducibly retrievable.
- [ ] Original papers, repositories, authors and licences or permissions are credited.
- [x] Study-specific modifications are distinguishable from upstream code.
- [ ] Paper configurations, seeds and evaluation commands are recorded.
- [ ] Dataset acquisition, preprocessing and redistribution constraints are documented.
- [x] Generated files, caches, checkpoints, compiled binaries and debug logs are excluded.
- [x] Conversion and evaluation code has focused automated tests.
- [x] Each integration has an evidence-backed verification status.
- [x] Known limitations and unverified paths are stated explicitly.
- [ ] The final paper artefact is tagged and archived at an immutable commit.

## Confirmed issues in the audited baseline

1. At the audited baseline, HGCLR and RADAr were described but absent. HGCLR has
   since been imported into the release-preparation branch; RADAr remains absent.
2. The repository contains a full copied PECOS tree. Its former submodule base
   and study modifications are now documented; broader reduction is deferred.
3. At the audited baseline there was no top-level licence, citation file,
   third-party notice or automated CI. A citation file, notice and bounded CI
   now exist; the study licence remains open.
4. At the audited baseline `XMLmodels/CascadeXML/requirements.txt` was empty. It
   now records the upstream-derived package inventory, explicitly labelled as
   reconstructed and not yet installation-verified.
5. `XMLmodels/CascadeXML/main_inference.py` imports the absent `Runner_accelerate` module.
6. The root README contains stale paths, including `XMLmodels/CascadeXML/src`.
7. Generated scheduler output, notebook checkpoints and a platform-specific compiled
   PECOS extension are tracked.
8. The baseline tracked corpus-derived HBGL files. Derived NYT text JSON and raw
   RCV1/WOS corpora have been removed; retained taxonomies and label vocabularies
   still require a redistribution review.
9. No fresh-clone end-to-end execution has yet been recorded for the current tree.

Resolution notes: the citation, third-party notice and CI portions of item 3 and
items 5-7 have been corrected on the release-preparation branch. CascadeXML's
missing environment remains open; its bounded upstream diff, entry-point fixes
and limitations are recorded in
[`CASCADEXML_MODIFICATIONS.md`](CASCADEXML_MODIFICATIONS.md).

The XR-Transformer former submodule base, post-conversion changes, corrected
preprocessing/evaluation paths and limitations are recorded in
[`XR_TRANSFORMER_MODIFICATIONS.md`](XR_TRANSFORMER_MODIFICATIONS.md). Its compact
historical aggregates remain unreconciled because several datasets contain
identical per-seed records.

## Scope boundaries

Release preparation should avoid broad refactoring of upstream model internals.
Confirmed blockers may be fixed, but working historical implementations should
otherwise be preserved. Effort should concentrate on:

- provenance and attribution;
- data conversion and hierarchy correctness;
- portable configurations and launch instructions;
- metric correctness;
- removal of accidental artefacts;
- bounded smoke tests;
- honest documentation of limitations.

## Historical implementation intake

Recovered working copies should be preserved before cleaning. For each available
integration, the release record should capture:

- Git remote, current commit and working-tree status, if available;
- a patch of uncommitted changes;
- relevant dependency and compatibility information;
- launch scripts and experiment configurations;
- locations or descriptions of paper-result evidence;
- a file inventory excluding datasets, checkpoints, caches and credentials.

Untouched working copies should remain available until their cleaned release
integrations have passed the selected verification level.

### HGCLR intake state

An independent audit was received on 2026-07-17 for the historical
`contrastive-htc` working checkout. It establishes the upstream reference,
modified tracked files, untracked source additions, environment adaptations and
excluded generated artefacts. The reviewed source-only intake has been imported
under `integrations/hgclr`.

Intake and reconciliation status:

- [x] preserve the tracked modifications for the four changed upstream files;
- [x] preserve the untracked source, documentation, aggregation and launch files;
- [x] exclude `checkpoints/` and generated `data/**/*.bin`, `data/**/*.idx` and
  `data/**/*.pt` files;
- [x] reconcile the final seed aggregates with the paper tables;
- [x] generalise machine-specific paths and remove account-specific details;
- [x] define and unit-test the empty-gold R-Precision policy;
- [ ] Deferred: the historical empty-gold audit is out of the current scope. The
  candidate aggregates were used for reconciliation during intake but are not
  published as release results.

The tracked patch and untracked release files were imported from the validated
intake material. The imported release code now defines R-Precision as invalid for
empty-gold samples and the preprocessing scripts enforce the same invariant.
Historical binaries would still need to be audited before the old aggregate
evidence could be treated as independently verified. This check does not block
the remaining repository clean-up.

### RADAr intake state

The implementation used in the study is currently unavailable. RADAr remains
credited and linked as an evaluated upstream method, but its study-specific
integration is deferred. The release must state this limitation rather than
substituting an unverified reconstruction.

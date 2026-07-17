# Model integration provenance

The models evaluated in this study were developed by their respective original
authors. This repository must not describe them as models introduced by this
project. It records the adaptations needed to evaluate them across HTC and XMC
domains.

Commit identifiers below are provenance references discovered during the
release-readiness audit. They are not yet claims about the exact base revision
of every historical experiment unless explicitly stated.

| Integration | Original repository | Current repository state | Verification | Provenance work remaining |
| --- | --- | --- | --- | --- |
| CascadeXML | `xmc-aalto/cascadexml` | Modified source present under `XMLmodels/CascadeXML` | Documented | Capture paper commands/environment and smoke-test a representative GPU path. |
| XR-Transformer | `amzn/pecos` | Full modified PECOS tree present under `XMLmodels/pecos` | Documented | Recover paper-run aggregates, record the final environment and smoke-test a representative GPU path; decide whether to reduce the copied tree after publication. |
| HBGL | `kongds/HBGL` | Modified source present under `htc/hbgl` | Documented | Settle redistribution permission, capture the final environment and smoke-test a representative GPU path. |
| HGCLR | `wzh9969/contrastive-htc` | Source-only working fork imported under `integrations/hgclr` | Documented | Run the deferred empty-gold audit on historical binaries, then complete a bounded fresh-environment smoke test. |
| RADAr | `yousef-younes/RADAr` | Working study copy is currently unavailable | Deferred | Credit and link upstream now; add the study integration later if it becomes available. |

## Known upstream references

### CascadeXML

- Repository: <https://github.com/xmc-aalto/cascadexml>
- Current upstream reference inspected on 2026-07-17:
  `ce701f688aaf5d5c8abe979d192f9c8f224aec90`
- No top-level licence was detected in the inspected upstream checkout.
- Seven retained Python files match the inspected commit byte-for-byte and the
  remaining source has a bounded, documented diff. This comparison establishes
  the inspected base despite the absence of preserved fork history.
- The detailed comparison, release fixes and result-affecting caveats are in
  [`CASCADEXML_MODIFICATIONS.md`](CASCADEXML_MODIFICATIONS.md).

### XR-Transformer / PECOS

- Repository: <https://github.com/amzn/pecos>
- Licence included in the copied tree: Apache-2.0.
- Git history shows that `XMLmodels/pecos` was previously a submodule pinned to
  `3fccd9af1b287c1cab96a7a16e93c2ff0bfbc903`, before being converted into a
  normal directory in commit `1606f625070fb1ea8977cc1aa610ffa9e6b2bada`.
- The historical base and all post-conversion changes have been bounded. The
  conversion commit cannot reveal whether its source submodule had uncommitted
  edits, so that limitation is retained explicitly.
- Study adaptations, release fixes and result-affecting caveats are documented
  in [`XR_TRANSFORMER_MODIFICATIONS.md`](XR_TRANSFORMER_MODIFICATIONS.md).
- The copied tree still contains substantial unrelated upstream material. A
  wider reduction is deferred to avoid destabilising the release integration.

### HBGL

- Repository: <https://github.com/kongds/HBGL>
- Current upstream reference inspected on 2026-07-17:
  `a40acdf87407a5a6cdd4c921c80c60b9f3522aa1`
- No top-level licence was detected in the inspected upstream checkout.
- The large textual diff is mostly line-ending conversion. The substantive
  refactor, cross-domain changes and fixes have been bounded against this commit
  and documented in [`HBGL_MODIFICATIONS.md`](HBGL_MODIFICATIONS.md).

### HGCLR

- Repository: <https://github.com/wzh9969/contrastive-htc>
- Working fork: historical `contrastive-htc` checkout, branch `main`.
- Upstream reference inspected and reported by both audits on 2026-07-17:
  `322a7ff2d83c878534bed25bb288cf4479d00363`
- Licence: MIT.
- The working tree had four modified tracked files (`eval.py`, `utils.py`,
  `train.py`, and `test.py`) plus source-only preprocessing, seed-sweep,
  aggregation and cluster-support additions.
- The modifications add P@1/P@3/P@5 and R-Precision, cost instrumentation,
  modern-stack compatibility, sequential five-seed execution and result
  aggregation.
- Generated checkpoints, binarised datasets and machine-local state were
  excluded from the public repository.
- The source changes were preserved during release intake before integration.

### RADAr

- Repository: <https://github.com/yousef-younes/RADAr>
- Current upstream reference inspected on 2026-07-17:
  `5cb2b785dd488cab422ac1d3a2d7744ed925c648`
- No top-level licence was detected in the inspected upstream checkout.
- Credit and redistribution permission should be recorded before the final
  public release.
- Upstream `requirements.txt` contains machine-specific `file:///` references;
  the release integration needs a portable environment specification.
- Access to the implementation used in the study is currently unavailable.
  RADAr integration work is therefore deferred and does not block the remaining
  repository clean-up.

## Required record for each integration

Before publication, each integration should document:

1. original paper and citation;
2. original repository and base commit;
3. original authors and applicable licence or permission;
4. files changed for this study and why;
5. datasets and configurations used in the paper;
6. environment and hardware assumptions;
7. verification evidence and known limitations.

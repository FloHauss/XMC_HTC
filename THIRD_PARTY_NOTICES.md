# Third-party notices

This repository integrates research software developed by other authors. The
cross-domain study does not claim ownership of the evaluated models. Users must
follow the terms of each upstream project and cite the corresponding papers.

## HGCLR

- Project: *Incorporating Hierarchy into Text Encoder: a Contrastive Learning
  Approach for Hierarchical Text Classification*
- Upstream: <https://github.com/wzh9969/contrastive-htc>
- Integrated base: `322a7ff2d83c878534bed25bb288cf4479d00363`
- Licence: MIT
- Retained licence: [`integrations/hgclr/LICENSE`](integrations/hgclr/LICENSE)

## XR-Transformer / PECOS

- Upstream: <https://github.com/amzn/pecos>
- Licence in the retained source tree: Apache-2.0
- Retained licence: [`XMLmodels/pecos/LICENSE`](XMLmodels/pecos/LICENSE)
- The repository contains a copied, study-adapted PECOS tree. Its former
  submodule revision, provenance limitation, adaptations and verification
  boundary are documented in
  [`XMLmodels/pecos/STUDY_INTEGRATION.md`](XMLmodels/pecos/STUDY_INTEGRATION.md).

## CascadeXML

- Upstream: <https://github.com/xmc-aalto/cascadexml>
- Inspected base: `ce701f688aaf5d5c8abe979d192f9c8f224aec90`
- No top-level licence was detected in that upstream revision. Redistribution
  permission remains to be settled.
- The retained code is an adapted research integration; its bounded provenance,
  changes and limitations are in
  [`XMLmodels/CascadeXML/README.md`](XMLmodels/CascadeXML/README.md).

## HBGL

- Upstream: <https://github.com/kongds/HBGL>
- Inspected base: `a40acdf87407a5a6cdd4c921c80c60b9f3522aa1`
- No top-level licence was detected in that upstream revision. Redistribution
  permission remains to be settled.
- The retained code is an adapted research integration and remains attributed
  to its original authors. It also derives from Microsoft UniLM `s2s-ft`, as
  stated by the upstream HBGL project; see [`htc/hbgl/README.md`](htc/hbgl/README.md).

## RADAr

- Upstream: <https://github.com/yousef-younes/RADAr>
- Upstream reference recorded for this study: `5cb2b785dd488cab422ac1d3a2d7744ed925c648`
- The repository includes [`htc/Radar++`](htc/Radar++/README.md), a local
  study adaptation of upstream RADAr for larger XML label spaces. It is not
  represented as an upstream RADAr release.
- No top-level licence was detected in the recorded upstream revision, and the
  local RADAr++ directory has no separate licence file. Permission or licence
  clarification is required before redistribution can be treated as settled.

## Datasets and other dependencies

Dataset licences and acquisition conditions are separate from model-code terms.
In particular, NYT and RCV1-V2 require users to follow their respective access
and redistribution conditions. Dependencies installed from PyPI, conda, system
packages or source repositories retain their own licences.

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
- The repository contains study-specific execution and evaluation changes. Their
  exact diff is still being isolated.

## CascadeXML

- Upstream: <https://github.com/xmc-aalto/cascadexml>
- No top-level licence was detected in the upstream revision inspected during
  release preparation.
- The retained code is an adapted research integration and remains attributed
  to its original authors.

## HBGL

- Upstream: <https://github.com/kongds/HBGL>
- No top-level licence was detected in the upstream revision inspected during
  release preparation.
- The retained code is an adapted research integration and remains attributed
  to its original authors. It also derives from Microsoft UniLM `s2s-ft`, as
  stated by the upstream HBGL project.

## RADAr

- Upstream: <https://github.com/yousef-younes/RADAr>
- The implementation used in the study is currently unavailable, so no RADAr
  source is redistributed in this release-preparation tree.
- The project works directly with the upstream owner and will preserve explicit
  credit if the study integration is added later.

## Datasets and other dependencies

Dataset licences and acquisition conditions are separate from model-code terms.
In particular, NYT and RCV1-V2 require users to follow their respective access
and redistribution conditions. Dependencies installed from PyPI, conda, system
packages or source repositories retain their own licences.

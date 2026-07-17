# On the Transferability Between Extreme Multi-Label and Hierarchical Text Classification

This repository accompanies our study of whether established extreme
multi-label classification (XML) and hierarchical text classification (HTC)
methods transfer effectively to datasets from the other domain.

We evaluate five existing methods - CascadeXML, XR-Transformer, HBGL, HGCLR and
RADAr - on three HTC datasets (WOS, NYT and RCV1-V2) and two XML datasets
(Wiki10-31K and AmazonCat-13K). The models were developed by their respective
original authors. Our contribution is the cross-domain evaluation, including
dataset conversion, model adaptation, evaluation and empirical comparison.

## At a glance

- [Read the paper](https://doi.org/10.1145/3820755.3832808)
- [Read the final results](RESULTS.md)
- [Convert datasets between HTC and XML formats](dataset_transfer/README.md)
- [Set up an individual model integration](#model-integrations)
- [See authors and acknowledgements](AUTHORS.md)
- [Cite the paper](#citation)

## Main findings

- XML methods transfer strongly to HTC datasets and are competitive with native
  HTC methods across ranking and F1 metrics.
- HTC methods are less effective on the XML datasets, particularly on ranking
  metrics.
- HGCLR could not complete training on the XML datasets because its dense label
  representations become prohibitively memory-intensive at XML label-set sizes.
- No single method dominates every dataset and metric: ranking and F1 performance
  often favour different model families.

The complete values, including deviations over runs, are presented in
[`RESULTS.md`](RESULTS.md).

## Model integrations

| Method | Original implementation | Study integration | Availability |
| --- | --- | --- | --- |
| CascadeXML | [xmc-aalto/cascadexml](https://github.com/xmc-aalto/cascadexml) | [`XMLmodels/CascadeXML`](XMLmodels/CascadeXML/README.md) | Included |
| XR-Transformer | [amzn/pecos](https://github.com/amzn/pecos) | [`XMLmodels/pecos`](XMLmodels/pecos/STUDY_INTEGRATION.md) | Included |
| HBGL | [kongds/HBGL](https://github.com/kongds/HBGL) | [`htc/hbgl`](htc/hbgl/README.md) | Included |
| HGCLR | [wzh9969/contrastive-htc](https://github.com/wzh9969/contrastive-htc) | [`integrations/hgclr`](integrations/hgclr/README.md) | Included |
| RADAr | [yousef-younes/RADAr](https://github.com/yousef-younes/RADAr) | Study code unavailable | Not included |

RADAr results are included in the paper and in [`RESULTS.md`](RESULTS.md), but
the implementation used for the study is not currently available in this
repository.

## Using the repository

There is no single environment for all five methods. The integrations retain
different research-code dependency stacks and should be set up separately:

- [CascadeXML setup and expected data](XMLmodels/CascadeXML/README.md)
- [XR-Transformer environment guide](xr_transformer_guide.md)
- [HBGL setup, preprocessing and launchers](htc/hbgl/README.md)
- [HGCLR setup and usage](integrations/hgclr/USAGE.md)
- [HTC/XML dataset conversion](dataset_transfer/README.md)

The repository does not provide a unified training API. It preserves the study
integrations and the commands needed to understand and reuse the adaptations.
More detailed environment notes are available in the
[reproducibility guide](docs/REPRODUCIBILITY.md).

## Data

The datasets are not redistributed as part of this repository. Users must obtain
the original WOS, NYT, RCV1-V2, Wiki10-31K and AmazonCat-13K data under the terms
of their respective providers.

The converters under [`dataset_transfer/`](dataset_transfer/README.md) translate
between the line-oriented XML representation and the JSON Lines HTC
representation used by the model integrations. Generated datasets, checkpoints,
caches and scheduler logs are excluded from Git.

## Citation

If you use this repository, please cite:

> Florian Hauss, Tom Speier, Nerijus Bertalis, Paul Granse, Ferhat Gül, Leon
> Menkel, David Schüler, Lukas Galke Poech, and Ansgar Scherp. *On the
> Transferability Between Extreme Multi-Label and Hierarchical Text
> Classification*. [doi:10.1145/3820755.3832808](https://doi.org/10.1145/3820755.3832808).

Florian Hauss and Tom Speier contributed equally to this research.
Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

Please also cite the original publication for each evaluated method. Upstream
repositories, licences and attribution are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Limitations

- RADAr study code is not currently available.
- The repository contains adapted research implementations with different
  environments; it is not a production model library.
- The complete copied PECOS tree is retained to avoid destabilising the
  XR-Transformer integration, so that directory contains more than the study
  itself requires.
- Some original per-seed experiment outputs are unavailable. The values in
  [`RESULTS.md`](RESULTS.md) are the authoritative paper results.

## Further documentation

The [documentation index](docs/README.md) separates practical guides, model
provenance and maintainer-facing release records.

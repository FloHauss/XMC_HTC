# On the Transferability Between Extreme Multi-Label and Hierarchical Text Classification

This repository accompanies our study of whether established extreme
multi-label classification (XML) and hierarchical text classification (HTC)
methods transfer effectively to datasets from the other domain.


## At a glance

- [Read the paper](https://doi.org/10.1145/3820755.3832808)
- [Read the final results](RESULTS.md)
- [Convert datasets between HTC and XML formats](dataset_transfer/README.md)
- [Set up an individual model integration](#model-integrations)
- [See authors and acknowledgements](AUTHORS.md)
- [Cite the paper](#citation)

The complete values, including deviations over runs, are presented in
[`RESULTS.md`](RESULTS.md).

## Model integrations

| Method | Original implementation | Study integration |
| --- | --- | --- |
| CascadeXML | [xmc-aalto/cascadexml](https://github.com/xmc-aalto/cascadexml) | [`XMLmodels/CascadeXML`](XMLmodels/CascadeXML/README.md) |
| XR-Transformer | [amzn/pecos](https://github.com/amzn/pecos) | [`XMLmodels/pecos`](XMLmodels/pecos/STUDY_INTEGRATION.md) |
| HBGL | [kongds/HBGL](https://github.com/kongds/HBGL) | [`htc/hbgl`](htc/hbgl/README.md) |
| HGCLR | [wzh9969/contrastive-htc](https://github.com/wzh9969/contrastive-htc) | [`htc/hgclr`](htc/hgclr/README.md) |
| RADAr (RADAr++ in this study) | [yousef-younes/RADAr](https://github.com/yousef-younes/RADAr) | [`htc/Radar++`](htc/Radar++/README.md) |

All evaluated models were introduced by their original authors. This repository
contains study-specific dataset conversion, adaptations, evaluation and the paper results.

## Using the repository

There is no single environment for all five methods. The integrations retain
different research-code dependency stacks and should be set up separately:

- [CascadeXML setup and expected data](XMLmodels/CascadeXML/README.md)
- [XR-Transformer environment guide](xr_transformer_guide.md)
- [HBGL setup, preprocessing and launchers](htc/hbgl/README.md)
- [HGCLR setup and usage](htc/hgclr/USAGE.md)
- [RADAr++ setup and usage](htc/Radar++/README.md)
- [HTC/XML dataset conversion](dataset_transfer/README.md)

The repository does not provide a unified training API. It preserves the study
integrations and the commands needed to understand and reuse the adaptations.


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


Please also cite the original publication for each evaluated method. Upstream
repositories, licences and attribution are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

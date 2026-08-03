# Results

The tables below reproduce the final results reported in *On the Transferability
Between Extreme Multi-Label and Hierarchical Text Classification*. Values are
shown as mean ± standard deviation over runs, in percentage points. The best
mean for each dataset and metric is shown in **bold**.

These tables are the authoritative result record for this repository. They
should not be replaced by older compact logs retained alongside individual
integrations.

## HTC datasets

### WOS

| Method | R-Prec | P@1 | P@3 | P@5 | F1-Micro | F1-Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XR-Transformer | 82.37 ± 1.42 | 89.94 ± 0.88 | 58.34 ± 0.93 | 36.29 ± 0.67 | 81.46 ± 1.83 | 69.80 ± 0.78 |
| CascadeXML | 86.25 ± 0.24 | 90.70 ± 0.15 | 59.61 ± 0.03 | 36.92 ± 0.01 | 86.70 ± 0.17 | 80.86 ± 0.25 |
| HGCLR | 86.39 ± 0.12 | 91.04 ± 0.15 | **60.06 ± 0.13** | **37.40 ± 0.11** | 86.82 ± 0.12 | 81.07 ± 0.24 |
| HBGL | 85.90 ± 0.15 | 88.30 ± 0.29 | 57.45 ± 0.11 | 34.48 ± 0.06 | **87.27 ± 0.15** | **81.87 ± 0.29** |
| RADAr | **86.60 ± 0.17** | **91.43 ± 0.10** | 57.73 ± 0.11 | 34.64 ± 0.06 | 86.60 ± 0.17 | 81.04 ± 0.26 |

### NYT

| Method | R-Prec | P@1 | P@3 | P@5 | F1-Micro | F1-Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XR-Transformer | 84.86 ± 1.53 | 94.76 ± 1.01 | 84.56 ± 0.78 | 72.33 ± 0.67 | 78.90 ± 0.75 | 66.83 ± 1.50 |
| CascadeXML | **85.31 ± 0.09** | **95.66 ± 0.29** | **84.77 ± 0.16** | **72.36 ± 0.11** | 79.67 ± 0.07 | 67.62 ± 0.12 |
| HGCLR | 84.07 ± 0.30 | 94.03 ± 0.42 | 83.53 ± 0.27 | 71.53 ± 0.22 | 78.55 ± 0.29 | 67.27 ± 0.44 |
| HBGL | 78.78 ± 0.16 | 82.14 ± 0.09 | 77.36 ± 0.21 | 65.82 ± 0.10 | **80.33 ± 0.14** | **69.73 ± 0.07** |
| RADAr | 79.74 ± 0.13 | 94.71 ± 0.08 | 81.27 ± 0.21 | 68.46 ± 0.07 | 79.04 ± 0.34 | 67.37 ± 0.92 |

### RCV1-V2

| Method | R-Prec | P@1 | P@3 | P@5 | F1-Micro | F1-Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XR-Transformer | 90.26 ± 0.56 | **97.35 ± 0.30** | **83.47 ± 0.26** | 57.22 ± 0.28 | 86.04 ± 0.31 | 64.43 ± 1.18 |
| CascadeXML | **90.45 ± 0.10** | 96.62 ± 0.16 | 83.38 ± 0.09 | **57.86 ± 0.04** | 86.65 ± 0.11 | 68.23 ± 0.08 |
| HGCLR | 90.15 ± 0.16 | 96.62 ± 0.31 | 83.22 ± 0.11 | 57.80 ± 0.06 | 85.98 ± 0.27 | 68.00 ± 0.50 |
| HBGL | 85.48 ± 0.16 | 92.90 ± 0.57 | 80.39 ± 0.25 | 54.32 ± 0.10 | 87.00 ± 0.05 | **70.48 ± 0.29** |
| RADAr | 86.80 ± 0.02 | 96.61 ± 0.09 | 81.46 ± 0.03 | 54.70 ± 0.03 | **87.26 ± 0.05** | 69.07 ± 0.54 |

Across the HTC datasets, the transferred XML methods remain competitive with
native HTC methods. CascadeXML is particularly strong on ranking metrics, while
HBGL and RADAr obtain several of the best F1 scores.

## XML datasets

### Wiki10-31K

| Method | R-Prec | P@1 | P@3 | P@5 | F1-Micro | F1-Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XR-Transformer | **38.12 ± 0.88** | **88.35 ± 0.29** | **79.66 ± 0.69** | **70.27 ± 0.85** | 15.12 ± 0.09 | 0.35 ± 0.02 |
| CascadeXML | 33.81 ± 0.06 | 87.02 ± 0.57 | 76.81 ± 0.01 | 66.53 ± 0.20 | **31.63 ± 0.79** | 3.24 ± 1.41 |
| HGCLR† | - | - | - | - | - | - |
| HBGL | 19.16 ± 0.20 | 67.53 ± 0.86 | 52.51 ± 0.64 | 46.96 ± 0.57 | 26.45 ± 0.21 | 1.95 ± 0.07 |
| RADAr | 26.40 ± 0.31 | 79.87 ± 0.26 | 53.74 ± 0.60 | 44.91 ± 0.81 | 26.99 ± 0.39 | **3.25 ± 0.20** |

### AmazonCat-13K

| Method | R-Prec | P@1 | P@3 | P@5 | F1-Micro | F1-Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XR-Transformer | **82.02 ± 0.75** | **96.54 ± 0.34** | **83.38 ± 0.28** | **67.76 ± 0.31** | 72.18 ± 0.83 | 34.98 ± 1.12 |
| CascadeXML | 81.56 ± 0.33 | 96.06 ± 0.43 | 82.72 ± 0.23 | 67.35 ± 0.29 | **75.37 ± 0.55** | **40.58 ± 0.14** |
| HGCLR† | - | - | - | - | - | - |
| HBGL | 64.96 ± 0.27 | 79.02 ± 0.17 | 72.04 ± 0.13 | 56.42 ± 0.27 | 70.64 ± 0.24 | 16.76 ± 0.38 |
| RADAr | 74.75 ± 0.41 | 95.14 ± 0.06 | 77.40 ± 0.17 | 61.26 ± 0.21 | 69.95 ± 0.27 | 33.33 ± 0.96 |

† HGCLR did not complete training. Its graph encoder stores label
representations in dense matrices, which becomes prohibitively memory-intensive
at XML label-set sizes.

On both XML datasets, XR-Transformer achieves the strongest ranking results and
CascadeXML the strongest Micro-F1. CascadeXML leads Macro-F1 on AmazonCat-13K,
while RADAr narrowly leads Macro-F1 on Wiki10-31K. Overall, the native XML
methods outperform the transferred HTC methods, especially on ranking metrics.

## Result sources

The compact transcription used to check these tables is stored in
[`results/paper/table_results.csv`](results/paper/table_results.csv). Retained
integration outputs are historical records and do not reliably reproduce every
final table entry; the tables in this file are authoritative.

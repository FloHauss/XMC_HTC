# Paper-result reconciliation

This record compares the author-supplied final paper tables with compact
historical outputs retained in the repository. It does not treat a historical
match as a fresh verification run.

## Authoritative paper record

The 25 method/dataset rows are transcribed in
[`results/paper/table_results.csv`](../results/paper/table_results.csv). The
transcription is tested for completeness, numeric ranges, failed-run markers and
the best-performing methods stated in the paper text.

## HGCLR

The five-seed JSON aggregates under `integrations/hgclr/results/candidate` match
the WOS, NYT and RCV1-V2 paper rows across all six metrics. Candidate means and
population standard deviations, converted to percentage points, differ from the
displayed table entries by no more than 0.01.

This establishes that these are the historical aggregates underlying, or
numerically consistent with, the paper table. They remain labelled candidate
because the historical empty-gold R-Precision audit was explicitly deferred and
they are not new runs from the release tree.

## XR-Transformer

| Dataset | Retained compact aggregate versus paper | Disposition |
| --- | --- | --- |
| AmazonCat-13K | R-Precision, P@1, P@3 and P@5 means match exactly; retained deviations are all zero while paper deviations are non-zero. | Mean provenance only; per-seed records are not the paper runs. |
| Wiki10-31K | All four retained ranking means differ from the paper. | Unreconciled legacy output. |
| WOS | All four retained ranking means differ materially from the paper. | Unreconciled legacy output. |
| NYT | All four retained ranking means differ from the paper. | Unreconciled legacy output. |
| RCV1-V2 | All four retained ranking means differ from the paper. | Unreconciled legacy output. |

The compact XR files do not include the paper's F1 values. Repeated per-seed
blocks in several files further prevent them from supporting the reported
deviations. The paper table is therefore the only release record for most XR
results until the original experiment archive is recovered.

The retained Amazon-670K file has no corresponding row in the supplied paper
tables and remains outside this reconciliation.

## CascadeXML, HBGL and RADAr

No compact per-seed aggregate files for these methods were found in the release
tree. Their paper rows have been preserved, but cannot yet be tied to local raw
or aggregate evidence. RADAr's study implementation is also unavailable.

## Remaining work

1. Recover paper-run XR per-seed outputs if available, especially the F1 values.
2. Locate or export compact CascadeXML and HBGL per-seed aggregates.
3. Add RADAr evidence only when the study implementation becomes available.
4. Keep all paper records distinct from future fresh-checkout smoke-test output.

# Paper table results

[`table_results.csv`](table_results.csv) is the machine-readable transcription
of the two final paper tables supplied by the authors. It is the authoritative
record of values reported in the paper, not output from a new release run.

Most readers should use the formatted, explanatory tables in
[`RESULTS.md`](../../RESULTS.md).

## Schema

- `domain` is the source dataset family: `HTC` or `XML`.
- All metric means and standard deviations are stored in percentage points,
  exactly as displayed in the paper tables.
- `status=did_not_complete` records HGCLR's failed XML-dataset training; its
  metric cells are intentionally empty.
- Boldface is not stored. Best values can be derived as the maximum complete
  mean for each dataset and metric.

The paper reports R-Precision, P@1, P@3, P@5, Micro-F1 and Macro-F1. A table
match establishes provenance for a historical aggregate; it does not establish
that the current repository can reproduce that result from a fresh checkout.

## Reconciliation summary

- HGCLR candidate aggregates for WOS, NYT and RCV1-V2 match all reported means
  and standard deviations within 0.01 percentage point when population standard
  deviation is used. They are paper-reconciled historical records, subject to
  the separately deferred empty-gold audit.
- XR-Transformer's retained AmazonCat-13K file matches the four paper ranking
  means, but its five repeated records and zero deviations do not match the
  paper deviations.
- XR-Transformer's retained WOS, NYT, RCV1 and Wiki10-31K aggregates do not
  match the paper values. They remain unreconciled legacy outputs and must not
  be cited as the source of the paper table.
- No compact CascadeXML, HBGL or RADAr per-seed aggregates were found in the
  release tree, so their paper values currently have no repository-side
  aggregate to compare.

Detailed evidence is in
[`docs/RESULT_RECONCILIATION.md`](../../docs/RESULT_RECONCILIATION.md).

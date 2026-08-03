# Paper table results

[`table_results.csv`](table_results.csv) is the machine-readable transcription 
of the two final paper tables supplied by the authors.

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

# Candidate historical aggregates

These files were exported from the working HGCLR server checkout on 2026-07-17.
They contain five-seed test, validation and cost aggregates for WebOfScience,
NYT and RCV1.

They have been reconciled against the author-supplied final paper tables. All
six reported means and population standard deviations for all three datasets
match within 0.01 percentage point. They remain historical candidate results,
not fresh release verification, because the historical empty-gold audit was
deferred. The cost measurements also predate the CUDA timing synchronisation
added during release hardening.

See [`docs/RESULT_RECONCILIATION.md`](../../../../docs/RESULT_RECONCILIATION.md)
for the comparison boundary.

# Release-preparation cleanup log

This log records categories of material removed from the working release tree.
It is intentionally concise: private execution paths, scheduler details and
derived-data sizes are not part of the public release record.

## 2026-07-17

Removed generated and environment-specific artifacts:

- raw scheduler/debug outputs and notebook-checkpoint duplicates;
- a platform-specific compiled PECOS extension;
- derived tokenised dataset artifacts;
- stale submodule declarations that no longer represented active Git
  submodules;
- server-environment exports and non-authoritative candidate cost aggregates.

The compact XR-Transformer `average.txt` summaries, aggregation/launch scripts,
source modifications and hierarchy metadata were retained. No broader PECOS
source reduction was attempted in this step.

The generic preprocessors, taxonomies and label vocabularies were retained.
Generated dataset JSON beneath HBGL preprocessing directories is now ignored to
prevent accidental recommits.

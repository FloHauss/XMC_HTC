# Provenance

The integration was reconstructed on 2026-07-17 from:

1. upstream `wzh9969/contrastive-htc` commit
   `322a7ff2d83c878534bed25bb288cf4479d00363`;
2. a binary patch of the server working tree's modifications to `eval.py`,
   `utils.py`, `train.py` and `test.py`;
3. the audited untracked preprocessing, aggregation, launch and documentation
   files from the working server checkout.

The source intake archive had SHA-256:

```text
eda8b6f236739820489626dae18923379f34c91fbc9fd0e677bef310bf5b3bc7
```

The archive's internal manifest verified successfully before import. It did not
contain the reported 14 GB checkpoint directory or approximately 770 MB of
generated binarised dataset files.

Release-hardening changes made after the intake are summarised in
[README.md](README.md) and remain visible in the release-preparation branch
history.

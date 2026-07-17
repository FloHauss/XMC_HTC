# CascadeXML study integration

This directory contains the CascadeXML implementation adapted for the
cross-domain HTC/XMC evaluation. CascadeXML was developed by Siddhant Kharbanda,
Atmadeep Banerjee, Erik Schultheis and Rohit Babbar; it is not a model introduced
by this project.

- Original repository: <https://github.com/xmc-aalto/cascadexml>
- Inspected base commit: `ce701f688aaf5d5c8abe979d192f9c8f224aec90`
- Original paper: *CascadeXML: Rethinking Transformers for End-to-end
  Multi-resolution Training in Extreme Multi-label Classification* (2022)
- Verification status: **Documented** - conversion helpers and source syntax are
  tested, but no fresh GPU training run has been completed from this checkout.

See [the modification record](../../docs/CASCADEXML_MODIFICATIONS.md) for the
bounded upstream diff, result-affecting changes, fixes, and known limitations.

## Expected data layout

Training reads `./data/<dataset>/` relative to the working directory. The
directory is expected to contain:

```text
train_raw_texts.txt
test_raw_texts.txt
Y.trn.txt or Y.trn.npz
Y.tst.txt or Y.tst.npz
train.txt
```

The label files contain comma-separated zero-based label IDs. `train.txt` is the
XML repository sparse feature format with a shape header. Generated tokenisation,
label, TF-IDF, graph, cluster and inverse-propensity files are intentionally
ignored by Git.

## Entry points

Run commands from this directory so the historical local imports and `./data`
path resolve correctly:

```bash
cd XMLmodels/CascadeXML
python main.py --help
python main_inference.py --help
```

The paper-specific dataset commands and a captured environment are not present
in the historical repository. Do not infer final paper settings from parser
defaults. These missing records are tracked as release limitations.

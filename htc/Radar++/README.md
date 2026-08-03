# RADAr++ study integration

The paper tables use the method name **RADAr**. Upstream RADAr is a
sequence-to-sequence method for hierarchical text classification developed by
its original authors. This directory, added in commit `c8c013e`, is the local
**RADAr++** study adaptation that extends the workflow to larger XML label
spaces; it is not claimed to be identical to an upstream RADAr release.

- Upstream repository: <https://github.com/yousef-younes/RADAr>
- Recorded upstream reference: `5cb2b785dd488cab422ac1d3a2d7744ed925c648`
- Study adaptation: local configs and preprocessing support HTC and larger XML
  label spaces, with RADAr and a RoBERTa baseline variant.
- Licensing: no top-level licence was detected in the recorded upstream
  revision, and this directory has no separate licence file. Permission or
  licensing status must be clarified before redistribution is treated as
  settled.
- Verification: the repository contains source and configuration files, but no
  clean-environment, GPU or end-to-end reproduction is recorded.

## Preprocessing

The integration follows the [HBGL](https://github.com/kongds/HBGL) dataset
format. Each dataset needs `taxonomy.txt`, `train.json`, `val.json` and
`test.json` (JSON Lines files historically named `.json`). Each taxonomy line
contains a label followed by its child labels, separated by tabs. For flat label
sets, every label must be a child of `Root`. A `taxonomy-synthetic.txt` file may
be supplied to inject synthetic hierarchy labels at runtime.

Each JSON Lines record has `token` (text) and `label` (a list of label strings).
A non-empty validation set is required. WOS (`wos`), NYT (`nyt`) and RCV1-V2
(`rcv1`) follow HBGL preprocessing. AmazonCat-13K (`ac13`), Wiki10-31K (`w31`),
Wiki-500K (`w500`), Amazon-670K (`a670`) and Amazon-3M (`a3000`) use the local
XML preprocessing workflow; source datasets are available through
[AttentionXML](https://github.com/yourh/AttentionXML). Place each input under
its dataset directory, with unprocessed XML data in that directory's
`preprocess/` subdirectory.

## Running the code

Install the dependencies from `requirements.txt`, then run from `src/`:

```bash
torchrun main.py <dataset_name> <config_file> <seed> <mode>
```

`dataset_name` must match a dataset and configuration directory;
`config_file` is its YAML configuration; and `mode` selects training,
inference, evaluation or the complete pipeline. The historical study seeds are
`5`, `11`, `912`, `211` and `1007`. Console output reports metrics; models and
results are written under the corresponding dataset's `saved_models` and
`saved_results` directories. See [config.md](config.md) for configuration
options.

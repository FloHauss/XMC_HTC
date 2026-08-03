# HTC/XMC dataset conversion

These study-owned converters translate between the JSON Lines representation
used by the HTC integrations and the plain text/sparse-label input used by the
XML integrations. They use only the Python standard library.

Dataset content is not included. Users must obtain the source datasets under
their applicable terms and place them in an input root outside Git or in one of
the ignored working directories.

## HTC JSON Lines format

Each record must contain a string `token` and a non-empty label list:

```json
{"token": "Example document", "label": ["parent", "child"]}
```

A taxonomy is tab-separated with one parent followed by its children:

```text
Root\tparent
parent\tchild
```

The converters reject cyclic taxonomies, malformed records, empty gold labels
and mismatched text/label counts.

## HTC to XML

Place the following files under `<input-root>/<dataset>/`:

```text
<dataset>.taxonomy
<dataset>_train.json
<dataset>_val.json
<dataset>_test.json
```

Then run:

```bash
cd dataset_transfer
python htc_to_xml.py wos \
  --input-root /path/to/input/htc \
  --output-root /path/to/output/xml
```

Training and validation records are combined because the historical XML
workflow has no separate validation input. Output label rows are
comma-separated zero-based IDs, matching the label syntax used by the release
CascadeXML and XR-Transformer preprocessors. `id_map.json` records the deterministic
case-insensitive label ordering.

Use `--leaves-only` to remove every selected label that is an ancestor - direct
or transitive - of another selected label:

```bash
python htc_to_xml.py wos --leaves-only \
  --input-root /path/to/input/htc \
  --output-root /path/to/output/xml
```

## XML to HTC

Place these files under `<input-root>/<dataset>/`:

```text
<dataset>.taxonomy
<dataset>_label_map.txt
<dataset>_train_labels.txt
<dataset>_train_texts.txt
<dataset>_test_labels.txt
<dataset>_test_texts.txt
```

Taxonomy and label files may contain numeric IDs referring to the zero-based
line positions in the label map. The converter restores ancestor labels and,
by default, excludes the synthetic `Root` label from examples.

```bash
python xml_to_htc.py wiki10-31k \
  --input-root /path/to/input/xml \
  --output-root /path/to/output/htc \
  --validation-fraction 0.2 \
  --random-seed 0
```

The train/validation split is deterministic and is made once over the complete
training set. The historical script silently truncated outputs to 30,000 train
and 5,000 validation records. This was not intentional in the paper experiments
and is treated as a converter defect. Truncation is now opt-in for explicitly
bounded experiments only:

```bash
python xml_to_htc.py wiki10-31k \
  --max-train 30000 --max-validation 5000
```

Record any truncation in the experiment metadata because it does not reproduce
the paper's intended full-dataset protocol.

## HTC leaf-only variant

To retain only the most specific selected labels while staying in HTC format:

```bash
python htc_to_htc_lite.py wos \
  --input-root /path/to/input/htc \
  --output-root /path/to/output/htc-lite
```

The output taxonomy contracts omitted intermediate nodes and connects each
retained node to its nearest retained descendants. All retained labels must be
reachable from `Root`; otherwise conversion fails rather than writing an
incomplete hierarchy.

## Reproducibility notes

- Input records keep their original order except for the deterministic
  XML-to-HTC train/validation assignment.
- Label IDs and taxonomy children are written deterministically.
- Newlines embedded in HTC text are replaced by spaces for line-oriented XML
  input.
- Generated outputs should not be committed unless they are demonstrably
  redistributable metadata rather than corpus content.

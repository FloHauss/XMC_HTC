#!/usr/bin/env python3
"""Convert HTC JSONL splits to the shared plain-text XML representation."""

import argparse
import json
from pathlib import Path

try:
    from .common import leaf_only, load_taxonomy, read_jsonl
except ImportError:  # Direct execution from dataset_transfer/.
    from common import leaf_only, load_taxonomy, read_jsonl


def _normalise_text(text):
    return " ".join(text.splitlines())


def _convert_records(records, hierarchy, leaves_only):
    converted = []
    known_labels = set(hierarchy)
    for children in hierarchy.values():
        known_labels.update(children)
    for index, record in enumerate(records):
        labels = list(dict.fromkeys(record["label"]))
        if leaves_only:
            unknown = set(labels) - known_labels
            if unknown:
                raise ValueError(f"Example {index} has labels absent from the taxonomy: {sorted(unknown)}")
            labels = leaf_only(labels, hierarchy)
        if not labels:
            raise ValueError(f"Example {index} has no labels after filtering")
        converted.append((_normalise_text(record["token"]), labels))
    return converted


def convert(dataset, input_root=Path("input/htc"), output_root=Path("output/htc"), leaves_only=False):
    input_dir = Path(input_root) / dataset
    suffix = "_leaves" if leaves_only else ""
    output_dir = Path(output_root) / f"{dataset}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = load_taxonomy(input_dir / f"{dataset}.taxonomy") if leaves_only else {}
    train = read_jsonl(input_dir / f"{dataset}_train.json")
    validation = read_jsonl(input_dir / f"{dataset}_val.json")
    test = read_jsonl(input_dir / f"{dataset}_test.json")

    converted_train = _convert_records(train + validation, hierarchy, leaves_only)
    converted_test = _convert_records(test, hierarchy, leaves_only)
    label_names = sorted(
        {label for _, labels in converted_train + converted_test for label in labels},
        key=lambda value: (value.casefold(), value),
    )
    label_to_id = {label: index for index, label in enumerate(label_names)}

    (output_dir / "id_map.json").write_text(
        json.dumps(label_to_id, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_split(output_dir, "train", converted_train, label_to_id)
    _write_split(output_dir, "test", converted_test, label_to_id)
    return output_dir


def _write_split(output_dir, split, records, label_to_id):
    texts = "".join(f"{text}\n" for text, _ in records)
    labels = "".join(
        ",".join(str(label_to_id[label]) for label in row_labels) + "\n"
        for _, row_labels in records
    )
    (output_dir / f"{split}_raw_texts.txt").write_text(texts, encoding="utf-8")
    (output_dir / f"{split}_labels.txt").write_text(labels, encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--input-root", type=Path, default=Path("input/htc"))
    parser.add_argument("--output-root", type=Path, default=Path("output/htc"))
    parser.add_argument("--leaves-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    convert(arguments.dataset, arguments.input_root, arguments.output_root, arguments.leaves_only)

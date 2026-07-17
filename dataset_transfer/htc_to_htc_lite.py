#!/usr/bin/env python3
"""Reduce HTC records and taxonomy to their most specific selected labels."""

import argparse
from pathlib import Path

try:
    from .common import (
        contract_taxonomy,
        leaf_only,
        load_taxonomy,
        read_jsonl,
        write_jsonl,
        write_taxonomy,
    )
except ImportError:  # Direct execution from dataset_transfer/.
    from common import (
        contract_taxonomy,
        leaf_only,
        load_taxonomy,
        read_jsonl,
        write_jsonl,
        write_taxonomy,
    )


def _filter_records(records, hierarchy):
    filtered, retained = [], set()
    for index, record in enumerate(records):
        labels = leaf_only(list(dict.fromkeys(record["label"])), hierarchy)
        if not labels:
            raise ValueError(f"Example {index} has no labels after leaf filtering")
        retained.update(labels)
        filtered.append({**record, "label": labels})
    return filtered, retained


def convert(dataset, input_root=Path("input/htc"), output_root=Path("output/htc_lite")):
    input_dir = Path(input_root) / dataset
    output_dir = Path(output_root) / f"{dataset}_lite"
    output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = load_taxonomy(input_dir / f"{dataset}.taxonomy")

    converted = {}
    retained = set()
    for split in ("train", "val", "test"):
        records = read_jsonl(input_dir / f"{dataset}_{split}.json")
        converted[split], split_labels = _filter_records(records, hierarchy)
        retained.update(split_labels)

    contracted = contract_taxonomy(hierarchy, retained)
    write_taxonomy(contracted, output_dir / f"{dataset}_lite.taxonomy")
    for split, records in converted.items():
        write_jsonl(records, output_dir / f"{dataset}_lite_{split}.json")
    return output_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--input-root", type=Path, default=Path("input/htc"))
    parser.add_argument("--output-root", type=Path, default=Path("output/htc_lite"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    convert(arguments.dataset, arguments.input_root, arguments.output_root)

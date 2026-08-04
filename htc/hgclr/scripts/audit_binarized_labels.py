"""Audit generated HGCLR label matrices without loading document text."""

import argparse
import json
from pathlib import Path

import numpy as np


# fairseq 0.10 imports aliases removed from modern NumPy.
_NUMPY_LEGACY_ALIASES = {
    "bool": bool,
    "complex": complex,
    "float": float,
    "int": int,
    "object": object,
    "str": str,
}
for _name, _type in _NUMPY_LEGACY_ALIASES.items():
    if _name not in np.__dict__:
        setattr(np, _name, _type)

from fairseq.data import data_utils  # noqa: E402


DEFAULT_DATASETS = ("WebOfScience", "nyt", "rcv1")


def audit_dataset(data_root, dataset_name):
    labels = data_utils.load_indexed_dataset(
        str(data_root / dataset_name / "Y"), None, "mmap"
    )
    cardinalities = [int(row.sum().item()) for row in labels]
    return {
        "dataset": dataset_name,
        "samples": len(cardinalities),
        "empty_gold_samples": sum(value == 0 for value in cardinalities),
        "minimum_label_cardinality": min(cardinalities) if cardinalities else None,
        "maximum_label_cardinality": max(cardinalities) if cardinalities else None,
        "mean_label_cardinality": (
            sum(cardinalities) / len(cardinalities) if cardinalities else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "datasets": [audit_dataset(args.data_root, name) for name in args.datasets]
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")

    if any(item["empty_gold_samples"] for item in report["datasets"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

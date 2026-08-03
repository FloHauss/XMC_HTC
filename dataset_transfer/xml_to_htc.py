#!/usr/bin/env python3
"""Convert plain-text XML splits and a taxonomy to HTC JSONL."""

import argparse
import random
from pathlib import Path

try:
    from .common import expand_ancestors, load_taxonomy, write_jsonl
except ImportError:  # Direct execution from dataset_transfer/.
    from common import expand_ancestors, load_taxonomy, write_jsonl


def _read_label_map(path):
    labels = Path(path).read_text(encoding="utf-8").splitlines()
    if not labels or any(not label for label in labels):
        raise ValueError(f"Label map is empty or contains blank labels: {path}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Label map contains duplicate labels: {path}")
    return labels


def _map_taxonomy(input_path, label_names):
    mapped_lines = []
    with Path(input_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            mapped = []
            for field in fields:
                if field.isdigit():
                    index = int(field)
                    if index >= len(label_names):
                        raise ValueError(
                            f"Taxonomy label {index} is outside the label map at line {line_number}"
                        )
                    mapped.append(label_names[index])
                else:
                    mapped.append(field)
            mapped_lines.append(mapped)
    return mapped_lines


def _write_mapped_taxonomy(lines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for fields in lines:
            stream.write("\t".join(fields) + "\n")


def _read_parallel(text_path, label_path, label_names, hierarchy, include_root=False):
    texts = Path(text_path).read_text(encoding="utf-8").splitlines()
    label_lines = Path(label_path).read_text(encoding="utf-8").splitlines()
    if len(texts) != len(label_lines):
        raise ValueError(
            f"Text/label count mismatch: {text_path} has {len(texts)}, "
            f"{label_path} has {len(label_lines)}"
        )
    if not texts:
        raise ValueError(f"Dataset split is empty: {text_path}")

    records = []
    for row, (text, label_line) in enumerate(zip(texts, label_lines)):
        values = label_line.replace(",", " ").split()
        if not values:
            raise ValueError(f"Example {row} has no labels in {label_path}")
        try:
            indices = [int(value) for value in values]
        except ValueError as error:
            raise ValueError(f"Non-integer label at {label_path}:{row + 1}") from error
        if min(indices) < 0 or max(indices) >= len(label_names):
            raise ValueError(f"Label outside [0, {len(label_names)}) at {label_path}:{row + 1}")
        labels = [label_names[index] for index in dict.fromkeys(indices)]
        labels = expand_ancestors(labels, hierarchy, include_root=include_root)
        records.append({"token": text, "label": labels, "doc_topic": [], "doc_keyword": []})
    return records


def _split_train_validation(records, fraction, seed):
    if not 0 <= fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    if fraction == 0:
        return list(records), []
    if len(records) < 2:
        raise ValueError("At least two training examples are required for a validation split")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(len(records) * fraction))
    validation_indices = set(indices[:validation_size])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


def convert(
    dataset,
    input_root=Path("input/xml"),
    output_root=Path("output/xml"),
    validation_fraction=0.2,
    random_seed=0,
    max_train=None,
    max_validation=None,
    include_root_label=False,
):
    input_dir = Path(input_root) / dataset
    output_dir = Path(output_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = input_dir / dataset

    label_names = _read_label_map(f"{prefix}_label_map.txt")
    mapped_taxonomy = _map_taxonomy(input_dir / f"{dataset}.taxonomy", label_names)
    taxonomy_path = output_dir / f"{dataset}.taxonomy"
    _write_mapped_taxonomy(mapped_taxonomy, taxonomy_path)
    hierarchy = load_taxonomy(taxonomy_path)

    train_records = _read_parallel(
        f"{prefix}_train_texts.txt",
        f"{prefix}_train_labels.txt",
        label_names,
        hierarchy,
        include_root_label,
    )
    test_records = _read_parallel(
        f"{prefix}_test_texts.txt",
        f"{prefix}_test_labels.txt",
        label_names,
        hierarchy,
        include_root_label,
    )
    train_records, validation_records = _split_train_validation(
        train_records, validation_fraction, random_seed
    )
    if max_train is not None:
        if max_train < 1:
            raise ValueError("max_train must be positive")
        train_records = train_records[:max_train]
    if max_validation is not None:
        if max_validation < 1:
            raise ValueError("max_validation must be positive")
        validation_records = validation_records[:max_validation]
    if not train_records:
        raise ValueError("Training split is empty after splitting/truncation")

    write_jsonl(train_records, output_dir / f"{dataset}_train.json")
    write_jsonl(validation_records, output_dir / f"{dataset}_val.json")
    write_jsonl(test_records, output_dir / f"{dataset}_test.json")
    return output_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--input-root", type=Path, default=Path("input/xml"))
    parser.add_argument("--output-root", type=Path, default=Path("output/xml"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--include-root-label", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    convert(
        arguments.dataset,
        arguments.input_root,
        arguments.output_root,
        arguments.validation_fraction,
        arguments.random_seed,
        arguments.max_train,
        arguments.max_validation,
        arguments.include_root_label,
    )

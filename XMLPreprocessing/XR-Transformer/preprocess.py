#!/usr/bin/env python3
"""Convert the study's plain-text format into XR-Transformer inputs."""

import argparse
from pathlib import Path

from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def read_text(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"No examples found in {path}")
    return lines


def read_labels(path):
    rows = []
    for row, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            raise ValueError(f"Example {row} has no gold labels in {path}")
        labels = [int(value) for value in line.split(",")]
        if min(labels) < 0:
            raise ValueError(f"Example {row} has a negative label in {path}")
        rows.append(labels)
    if not rows:
        raise ValueError(f"No labels found in {path}")
    return rows


def create_label_matrix(rows, num_labels):
    row_indices, col_indices = [], []
    for row, labels in enumerate(rows):
        if max(labels) >= num_labels:
            raise ValueError(f"Example {row} has a label outside [0, {num_labels})")
        row_indices.extend([row] * len(labels))
        col_indices.extend(labels)
    return csr_matrix(
        ([1.0] * len(row_indices), (row_indices, col_indices)),
        shape=(len(rows), num_labels),
    )


def convert(input_dir, output_dir):
    train_text = read_text(input_dir / "train_raw_texts.txt")
    test_text = read_text(input_dir / "test_raw_texts.txt")
    train_labels = read_labels(input_dir / "train_labels.txt")
    test_labels = read_labels(input_dir / "test_labels.txt")
    if len(train_text) != len(train_labels):
        raise ValueError("Training text and label counts differ")
    if len(test_text) != len(test_labels):
        raise ValueError("Test text and label counts differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "tfidf-attnxml"
    feature_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(stop_words="english")
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)
    save_npz(feature_dir / "X.trn.npz", train_features.tocsr())
    save_npz(feature_dir / "X.tst.npz", test_features.tocsr())

    num_labels = max(
        max(label for row in train_labels for label in row),
        max(label for row in test_labels for label in row),
    ) + 1
    save_npz(output_dir / "Y.trn.npz", create_label_matrix(train_labels, num_labels))
    save_npz(output_dir / "Y.tst.npz", create_label_matrix(test_labels, num_labels))
    (output_dir / "X.trn.txt").write_text("\n".join(train_text) + "\n", encoding="utf-8")
    (output_dir / "X.tst.txt").write_text("\n".join(test_text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

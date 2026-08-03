"""Dependency-light file conversion helpers for CascadeXML datasets."""

import os

import scipy.sparse as sp


def make_csr_tfidf(dataset, _lf_data=False):
    file_name = f"{dataset}/tfidf.npz"
    if os.path.exists(file_name):
        print(f"Loading {file_name}")
        return sp.load_npz(file_name)

    with open(f"{dataset}/train.txt") as source:
        # XML repository train.txt files start with a shape header.
        lines = source.readlines()[1:]

    row_idx, col_idx, val_idx = [], [], []
    for row, line in enumerate(lines):
        for tfidf in line.split()[1:]:
            try:
                token, weight = tfidf.split(":")
            except ValueError:
                print(f"Issue with token at line number {row}: {tfidf}")
                continue
            row_idx.append(row)
            col_idx.append(int(token))
            val_idx.append(float(weight))

    if not row_idx:
        raise ValueError(f"No TF-IDF features found in {dataset}/train.txt")
    matrix = sp.csr_matrix(
        (val_idx, (row_idx, col_idx)),
        shape=(len(lines), max(col_idx) + 1),
    )
    print(f"Created {file_name}")
    sp.save_npz(file_name, matrix)
    return matrix


def make_csr_labels(num_labels, file_name, lf_data):
    if os.path.exists(file_name):
        print(f"Loading {file_name}")
        return sp.load_npz(file_name)

    with open(os.path.splitext(file_name)[0] + ".txt") as source:
        lines = source.readlines()
    if lf_data:
        lines = lines[1:]

    row_idx, col_idx = [], []
    for row, line in enumerate(lines):
        label_field = line.split()[0] if lf_data else line.strip()
        labels = [int(label) for label in label_field.split(",") if label]
        if not labels:
            raise ValueError(f"Example {row} has no gold labels in {file_name}")
        if min(labels) < 0 or max(labels) >= num_labels:
            raise ValueError(
                f"Example {row} contains a label outside [0, {num_labels})"
            )
        col_idx.extend(labels)
        row_idx.extend([row] * len(labels))

    matrix = sp.csr_matrix(
        ([1] * len(row_idx), (row_idx, col_idx)),
        shape=(len(lines), num_labels),
    )
    print(f"Created {file_name}")
    sp.save_npz(file_name, matrix)
    return matrix

#!/usr/bin/env python3

import argparse

import numpy as np

from pecos.utils.smat_util import CsrEnsembler, load_matrix, sorted_csr


def sparse_f1(y_true, y_pred):
    """Return micro- and macro-F1 from sparse binary label support."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")
    if y_true.shape[0] == 0 or y_true.shape[1] == 0:
        raise ValueError("evaluation matrices must be non-empty")

    true = y_true.copy().tocsr()
    pred = y_pred.copy().tocsr()
    true.eliminate_zeros()
    pred.eliminate_zeros()
    true.data = np.ones_like(true.data, dtype=np.uint8)
    pred.data = np.ones_like(pred.data, dtype=np.uint8)

    true_per_label = np.asarray(true.sum(axis=0)).ravel()
    pred_per_label = np.asarray(pred.sum(axis=0)).ravel()
    tp_per_label = np.asarray(true.multiply(pred).sum(axis=0)).ravel()

    denominator = true_per_label + pred_per_label
    per_label = np.divide(
        2.0 * tp_per_label,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    micro_denominator = true.nnz + pred.nnz
    micro = 2.0 * tp_per_label.sum() / micro_denominator if micro_denominator else 0.0
    return float(micro), float(per_label.mean())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--truth-path", required=True, metavar="PATH")
    parser.add_argument("-p", "--pred-path", required=True, nargs="+", metavar="PATH")
    parser.add_argument("--tags", required=True, nargs="+", metavar="TAG")
    parser.add_argument("--ens-method", default="rank_average")
    return parser


def do_evaluation(args):
    if len(args.tags) != len(args.pred_path):
        raise ValueError("--tags and --pred-path must have the same length")
    y_true = sorted_csr(load_matrix(args.truth_path).tocsr())
    predictions = [
        sorted_csr(load_matrix(path).tocsr()) for path in args.pred_path
    ]

    print("==== P@k, recall@k and R-Precision ====")
    CsrEnsembler.print_ens(
        y_true, predictions, args.tags, ens_method=args.ens_method
    )

    print("\n==== F1 over stored prediction support ====")
    for tag, prediction in zip(args.tags, predictions):
        micro, macro = sparse_f1(y_true, prediction)
        print(f"[{tag}] F1-micro: {micro:.4f}, F1-macro: {macro:.4f}")

    ensemble = getattr(CsrEnsembler, args.ens_method)(*predictions)
    micro, macro = sparse_f1(y_true, ensemble)
    print(
        f"[Ensemble-{args.ens_method}] F1-micro: {micro:.4f}, "
        f"F1-macro: {macro:.4f}"
    )


if __name__ == "__main__":
    do_evaluation(parse_arguments().parse_args())

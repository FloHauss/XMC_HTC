#!~/miniconda3/envs/xr_transformer_env/bin/python -u

from pecos.utils.smat_util import sorted_csr, CsrEnsembler, load_matrix
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-y",
        "--truth-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the file of with ground truth output (CSR: nr_insts * nr_items)",
    )
    parser.add_argument(
        "-p",
        "--pred-path",
        type=str,
        required=True,
        nargs="*",
        metavar="PATH",
        help="path to the file of predicted output (CSR: nr_insts * nr_items)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        nargs="*",
        metavar="PATH",
        help="tags attached to each prediction",
    )
    parser.add_argument(
        "--ens-method",
        type=str,
        metavar="STR",
        default="rank_average",
        help="prediction ensemble method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="PATH",
        help="path to dataset",
    )

    return parser


def do_evaluation(args):
    """ Evaluate xlinear predictions """
    assert len(args.tags) == len(args.pred_path)
    Y_true = sorted_csr(load_matrix(args.truth_path).tocsr())
    Y_pred = [sorted_csr(load_matrix(pp).tocsr()) for pp in args.pred_path]

    #if hasattr(args, "threshold") and args.threshold is not None:
        #thr = float(args.threshold)
        #print(f"Applying threshold: keeping labels with score ≥ {thr}")
        #Y_pred = [Yp.multiply(Yp >= thr).astype(bool).astype(int) for Yp in Y_pred]
        #for Yp in Y_pred:
            #Yp.eliminate_zeros()
    print("==== evaluation results ====")
    
    CsrEnsembler.print_ens(Y_true, Y_pred, args.tags, ens_method=args.ens_method)

    print("\n==== F1-scores ====")
    for tag, Yp in zip(args.tags, Y_pred):
        # Convert sparse matrices to binary dense arrays for sklearn (if feasible)
        y_true_bin = (Y_true > 0).astype(int).toarray()
        if args.threshold is not None:
            thr = float(args.threshold)
            y_pred_bin = (Yp >= thr).astype(int).toarray()
        else:
            y_pred_bin = (Yp > 0).astype(int).toarray()
        
        f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
        f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
        
        print(f"[{tag}] F1-micro: {f1_micro:.4f}, F1-macro: {f1_macro:.4f}")

    if args.ens_method is not None and len(Y_pred) > 1:
        if args.ens_method.lower() == "softmax_average":
            # Average scores element-wise
            Y_ens = sum(Y_pred) / len(Y_pred)
        elif args.ens_method.lower() in ["max", "vote"]:
            # Take element-wise maximum (equivalent to union of predicted labels)
            Y_ens = Y_pred[0].copy()
            for Yp in Y_pred[1:]:
                Y_ens = Y_ens.maximum(Yp)
        else:
            print(f"Warning: Unknown ensemble method '{args.ens_method}', skipping ensemble F1.")
            return

        # Binarize for F1 computation
        y_true_bin = (Y_true > 0).astype(int).toarray()
        y_ens_bin = (Y_ens > 0).astype(int).toarray()

        f1_micro_ens = f1_score(y_true_bin, y_ens_bin, average='micro', zero_division=0)
        f1_macro_ens = f1_score(y_true_bin, y_ens_bin, average='macro', zero_division=0)

        print(f"[Ensemble-{args.ens_method}] F1-micro: {f1_micro_ens:.4f}, F1-macro: {f1_macro_ens:.4f}")
        
if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_evaluation(args)


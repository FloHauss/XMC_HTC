#!~/miniconda3/envs/xr_transformer_env/bin/python -u

import argparse
import os
import numpy as np

from pecos.utils.smat_util import sorted_csr, CsrEnsembler, load_matrix

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
    print("==== evaluation results ====")
    CsrEnsembler.print_ens(Y_true, Y_pred, args.tags, ens_method=args.ens_method)

    # Deprecated. Alternative including psp@k
    # inv_prop = get_inv_prop(args.dataset)
    # CsrEnsembler.print_ens(Y_true, Y_pred, args.tags, prop_scores=inv_prop, ens_method=args.ens_method)


# Deprecated. Method
# def get_inv_prop(path_to_dataset):
#     inv_prop = np.load(os.path.join(path_to_dataset, 'inv_prop.npy'))
#     return inv_prop

if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_evaluation(args)


#!/usr/bin/env python3 -u

import argparse
import os
import numpy as np
import scipy.sparse as sp

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
        required=True
        metavar="PATH",
        help="path to dataset",
    )

    return parser


def do_evaluation(args):
    """ Evaluate xlinear predictions """
    assert len(args.tags) == len(args.pred_path)
    Y_true = sorted_csr(load_matrix(args.truth_path).tocsr())
    Y_pred = [sorted_csr(load_matrix(pp).tocsr()) for pp in args.pred_path]
    inv_prop = get_inv_prop(args.dataset)
    print("==== evaluation results ====")
    CsrEnsembler.print_ens(Y_true, Y_pred, args.tags, prop_scores=inv_prop ens_method=args.ens_method)

# def make_csr_labels(num_labels, Yt_path):
#     if os.path.exists(Yt_path):
#         print(f"Loading {Yt_path}")
#         Y = sp.load_npz(Yt_path)
#     else:
#         with open(os.path.splitext(Yt_path)[0]+'.txt') as fil:
#             row_idx, col_idx = [], []
#             for i, lab in enumerate(fil.readlines()):
#                 l_list = [int(l) for l in lab.replace('\n', '').split(',')]
#                 col_idx.extend(l_list)
#                 row_idx.extend([i]*len(l_list))

#             m = max(row_idx) + 1
#             n = num_labels
#             val_idx = [1]*len(row_idx)
#             Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
#             print(f"Created {Yt_path}")
#             sp.save_npz(Yt_path, Y)
#     return Y

# def get_inv_prop(path_to_dataset, Yt_path):
#     if os.path.exists(os.path.join(path_to_dataset, 'inv_prop.npy')):
#         inv_prop = np.load(os.path.join(path_to_dataset, 'inv_prop.npy'))
#         return inv_prop

#     d = path_to_dataset.split('/')[-1]

#     # num_labels = {'Wiki10-31K': 30938, 'AmazonCat-13K': 13330, 'Wiki-500K':501070, 'Amazon-670K': 670091, 'Amazon-3M': 2812281}
#     Y_train = 

#     print("Creating inv_prop file")
    
#     A = {'Eurlex': 0.6, 'LF-Amazon-131K': 0.6, 'Amazon-670K': 0.6, 'Amazon-3M': 0.6, 'AmazonCat-13K': 0.55, 'Wiki-500K' : 0.5, 'Wiki10-31K' : 0.55}
#     B = {'Eurlex': 2.6, 'LF-Amazon-131K': 2.6, 'Amazon-670K': 2.6, 'Amazon-3M': 2.6, 'AmazonCat-13K': 1.5, 'Wiki-500K': 0.4, 'Wiki10-31K': 1.5}
 
#     a, b = A[d], B[d]
    
#     num_samples = Y_train.shape[0]
#     inv_prop = np.array(Y_train.sum(axis=0)).ravel()
    
#     c = (np.log(num_samples) - 1) * np.power(b+1, a)
#     inv_prop = 1 + c * np.power(inv_prop + b, -a)
    
#     np.save(os.path.join(path_to_dataset, 'inv_prop.npy'), inv_prop)
#     return inv_prop

def get_inv_prop(path_to_dataset):
    inv_prop = np.load(os.path.join(path_to_dataset, 'inv_prop.npy'))
    return inv_prop

if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_evaluation(args)


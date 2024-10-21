from __future__ import absolute_import, division, print_function

import logging
import os
import json

from main.get_args import get_args
from main.get_model_and_tokenizer import get_model_and_tokenizer
from main.prepare import prepare
from main.tester import tester
from main.train import train

import torch

import wandb

from s2s_ft import utils

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGLEVEL)


def main():
    args = get_args()
    prepare(args, logger=logger)
    if args.only_test:
        args.wandb = False
        tester(args, args.only_test_path, None)
        exit(0)

    if args.wandb:
        wandb.init(
            project="HBGL",
            name=args.output_dir.split('/')[-1],
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer, vs = get_model_and_tokenizer(args, logger=logger)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        if not args.lmdb_cache:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training.pt")
        else:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training_lmdb")

    if args.soft_label:
        args.cached_train_features_file += 'soft_label'
        # args.valid_file = args.valid_file.replace('generated', 'generated_tl')
        #
        if args.soft_label_hier_real:
            hier_labels = None
            for line in open(args.train_file):
                if hier_labels:
                    for i, l in enumerate(json.loads(line)['tgt']):
                        hier_labels[i] |=  set(l)
                else:
                    hier_labels = [set(i) for i in json.loads(line)['tgt']]
            hier_labels = [tokenizer.convert_tokens_to_ids(list([j.lower() for j in i])) for i in hier_labels]

            def to_multi_hot(label):
                _label = torch.zeros(model.config.vocab_size)
                for i in label:
                    _label[i] = 1
                return _label.bool()

            model.hier_labels = [to_multi_hot(i) for i in hier_labels]
            model.soft_label_hier_real = args.soft_label_hier_real


    num_lines = sum(1 for line in open(args.train_file))
    training_features = utils.load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=True,
        lmdb_cache=args.lmdb_cache, lmdb_dtype=args.lmdb_dtype,
        soft_label=args.soft_label,
    )

    if args.add_vocab_file:
        for i in training_features:
            for j in i.target_ids:
                if args.soft_label:
                    for ji in j:
                        assert ji >= vs
                else:
                    j >= vs

    best_macro_f1_path, best_micro_f1_path = train(args, training_features, model, tokenizer, logger=logger)
    if args.test_file:
        tester(args, best_macro_f1_path, best_micro_f1_path)


if __name__ == "__main__":
    main()

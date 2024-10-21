import os
import wandb

def tester(args, best_macro_f1_path, best_micro_f1_path):
    from test import main
    bout = None
    for i, save_path in enumerate([best_micro_f1_path, best_macro_f1_path]):
        if save_path is None: continue
        flags = ['--model_type'     , args.model_type                          ,
            '--tokenizer_name'         , args.model_name_or_path             ,
            '--input_file'             , args.test_file                  ,
            '--split'                  , 'test'                         ,
            '--do_lower_case'          ,
            '--model_path'             , str(save_path)              ,
            '--max_seq_length'         , str(args.max_source_seq_length + args.max_target_seq_length) if args.label_cpt_decodewithpos else str(args.max_source_seq_length)             ,
            '--max_tgt_length'         , str(args.max_target_seq_length)             ,
            '--batch_size'             , '128'                            ,
            '--beam_size'              , '1'                             ,
            '--length_penalty'         , '0'                             ,
            '--forbid_duplicate_ngrams',
            '--mode'                   , 's2s'                           ,
            '--forbid_ignore_word'     , '"."'                           ,
            '--cached_features_file'   , str(os.path.join(args.output_dir, "cached_features_for_test.pt")),
            '--add_vocab_file'         , args.add_vocab_file]

        if args.softmax_label_only:
            flags.append('--softmax_label_only')
        if args.soft_label:
            flags.append('--soft_label')
        if args.soft_label_hier_real:
            flags.append('--soft_label_hier_real_with_train_file')
            flags.append(args.train_file)
        if args.model_type == 'roberta':
            del flags[flags.index('--do_lower_case')]
        if args.label_cpt_decodewithpos:
            flags.append('--target_no_offset')
        if args.ignore_meta_label:
            flags.append('--ignore_meta_label')

        out = main(flags)
        prefix = 'test' + 'micro' if i == 0 else 'macro'
        if args.wandb:
            wandb.log({f'{prefix}/macro_f1': out['macro_f1'], f'{prefix}/micro_f1': out['micro_f1']})
            if bout is None or bout['macro_f1'] < out['macro_f1']:
                bout = out

    if args.wandb and bout:
        prefix = 'test'
        wandb.log({f'{prefix}/macro_f1': bout['macro_f1'], f'{prefix}/micro_f1': bout['micro_f1']})
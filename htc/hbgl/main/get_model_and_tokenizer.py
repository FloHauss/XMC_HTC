import os
import logging

from main.training_cpt import training_cpt
import torch

from s2s_ft.modeling import BertForSequenceToSequenceWithPseudoMask, BertForSequenceToSequenceUniLMV1
from transformers import BertConfig, BertTokenizer

from s2s_ft.config import BertForSeq2SeqConfig

import main.tree_split as ts

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
}

def get_model_and_tokenizer(args, logger):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        fix_word_embedding=args.fix_word_embedding,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logger.info("Model config for seq2seq: %s", str(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)

    model_class = \
        BertForSequenceToSequenceWithPseudoMask if args.mask_way == 'v2' \
            else BertForSequenceToSequenceUniLMV1

    logger.info("Construct model %s" % model_class.MODEL_NAME)

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, model_type=args.model_type,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.add_vocab_file:
        import pickle
        with open(args.add_vocab_file, 'rb') as f:
            label_map = pickle.load(f)
        ##### print(f'label_map: {label_map}')
        label_tokens_start_index  = model.bert.embeddings.word_embeddings.num_embeddings
        labels_key = list(label_map.keys())
        label_name_tensors = []
        max_l = -1
        if args.rcv1_expand:
            rcv1_label_expand = {}
            for i in open(args.rcv1_expand):
                oi = [j for j in i.replace('\n', '').split(' ') if len(j) > 0]
                rcv1_label_expand[oi[3]] = i.split('child-description: ')[-1].lower().replace('\n', '')

        for lk in labels_key:
            if args.one_by_one_label_init_map:
                from collections import defaultdict
                hiera = defaultdict(set)
                _label_dict = {}
                with open(args.one_by_one_label_init_map) as f:
                    _label_dict['Root'] = -1
                    for line in f.readlines():
                        line = line.strip().split('\t')
                        for i in line[1:]:
                            if i not in _label_dict:
                                _label_dict[i] = len(_label_dict) - 1
                            hiera[line[0]].add(i)
                    _label_dict.pop('Root')

                r_hiera = {}
                for i in hiera:
                    for j in list(hiera[i]):
                        r_hiera[j] = i

                def _loop(a):
                    if r_hiera[a] != 'Root':
                        return [a,] + _loop(r_hiera[a])
                    else:
                        return [a]

                one_by_one_label_init_map = {}
                for i in _label_dict:
                    one_by_one_label_init_map[i] = '/'.join(_loop(i)[::-1])
                print(f'map {lk} to {one_by_one_label_init_map[lk]}')
                label_name_tensors.append(tokenizer.encode(one_by_one_label_init_map[lk], add_special_tokens=False))
            elif args.nyt_only_last_label_init:
                print(f'map {lk} to {lk.split("/")[-1]}')
                label_name_tensors.append(tokenizer.encode(lk.split("/")[-1], add_special_tokens=False))
            elif args.rcv1_expand:
                print(f'map {lk} to {rcv1_label_expand[lk]}')
                label_name_tensors.append(tokenizer.encode(rcv1_label_expand[lk], add_special_tokens=False))
            else:
                label_name_tensors.append(tokenizer.encode(lk, add_special_tokens=False))
            max_l = max(len(label_name_tensors[-1]), max_l)
        label_name_tensors = torch.LongTensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors])

        with torch.no_grad():
            init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
            label_mask = label_name_tensors != tokenizer.pad_token_id
            init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1)
        label_tokens = [i for i in range(len(label_map))]
        tokenizer.add_tokens([label_map[label] for label in labels_key])
        #import pdb;pdb.set_trace()
        #labels_embeds = torch.nn.Embedding(len(label_tokens), config.hidden_size).weight.data
        if args.label_cpt:
            # for compare with same seed
            rng_state = torch.get_rng_state()

            from collections import defaultdict
            hiera = defaultdict(set)
            _label_dict = {}
            with open(args.label_cpt) as f:
                _label_dict['Root'] = -1
                for line in f.readlines():
                    line = line.strip().split('\t')
                    for i in line[1:]:
                        if i not in _label_dict:
                            _label_dict[i] = len(_label_dict) - 1
                        hiera[line[0]].add(i)
                _label_dict.pop('Root')
            r_hiera = {}
            for i in hiera:
                for j in list(hiera[i]):
                    r_hiera[j] = i

            def _loop(a):
                if r_hiera[a] != 'Root':
                    return [a,] + _loop(r_hiera[a])
                else:
                    return [a]

            label_class = {}
            for i in _label_dict:
                label_class[i] = len(_loop(i))
            # cls l1 l2 l3 sep
            attention_mask = torch.zeros((len(label_tokens) + 2, len(label_tokens) + 2))
            num_hiers = defaultdict(set)
            reversed_hiers = {}
            for hi in hiera:
                for hj in list(hiera[hi]):
                    def _label_map_f(x):
                        if x == 'Root': return -1
                        return int(label_map[x].replace('[A_', '').replace(']', ''))
                    attention_mask[_label_map_f(hi) + 1][_label_map_f(hj) + 1] = 1
                    num_hiers[_label_map_f(hi) + 1].add(_label_map_f(hj) + 1)
                    reversed_hiers[_label_map_f(hj) + 1] = _label_map_f(hi) + 1

                    if args.label_cpt_use_bce:
                        attention_mask[_label_map_f(hj) + 1][_label_map_f(hi) + 1] = 1
                    if args.self_attention:
                        attention_mask[_label_map_f(hi) + 1][_label_map_f(hi) + 1] = 1
            input_ids = torch.LongTensor(tokenizer.encode(' '.join(label_map.values()).lower()))
            cls = input_ids[0]
            assert len(input_ids) == len(labels_key) + 2
            position_ids = torch.LongTensor([0, ] + [label_class[i] for i in labels_key] + [max(label_class.values()) + 1,])

            ### CUSTOM
            leaf_trees = ts.find_leaf_trees('Root', hiera)
            hierarchies = ts.k_merger(500, leaf_trees)

            SPLIT_ids = []
            SPLIT_input_ids = []
            SPLIT_attention_masks = []
            SPLIT_position_ids = []
            SPLIT_init_label_emb = []
            SPLIT_num_hiers = []
            SPLIT_reversed_hiers = []
            SPLIT_pos_to_idx = []

            label_to_pos = {label: int(label_map[label].replace('[A_', '').replace(']', '')) + 1 for label in label_map}
            label_to_pos['Root'] = 0
            #r_id_hiera = {label_to_id[child]: [label_to_id[parent]] for child, parent in r_hiera.items()}
            r_id_hiera = {label_to_pos[child]: label_to_pos[parent] for child, parent in r_hiera.items()}

            #print(input_ids)
            for hierarchy in hierarchies:
                labels = ts.flatten_tree(hierarchy)
                #print(labels)
                labels.remove('Root')

                #ids = [0] + [label_to_idx[label] for label in labels] + [max(label_class.values()) + 1]
                ids = [label_to_pos[label] for label in labels] # perverse
                ids.sort()
                ids = [0] + ids + [-1]
                #print('ids:')
                #print(ids)
                SPLIT_ids.append(ids)

                sub_attention_mask = attention_mask[ids][:,ids]
                #sub_input_ids = torch.cat((torch.unsqueeze(input_ids[0], dim=0), input_ids[ids], torch.unsqueeze(input_ids[-1], dim=0)))
                sub_input_ids = input_ids[ids]
                sub_position_ids = position_ids[ids]
                sub_init_label_emb = init_label_emb[[id - 1 for id in ids[1:-1]]] # ignore CLS and SEP embedding, shift index one to the left for zero-based access
                
                #idx = [id - 1 for id in ids[1:-1]]
                pos_to_idx = {num: index for index, num in enumerate(ids)}
                #print('pos_to_idx:')
                #print(pos_to_idx)

                #sub_num_hiers = {pos_to_idx[label_to_pos[parent]]: [pos_to_idx[label_to_pos[child]] for child in children] for parent, children in hierarchy.items()}
                #sub_num_hiers = {label_to_pos[parent]: {label_to_pos[child] for child in children} for parent, children in hierarchy.items()}
                sub_num_hiers = defaultdict(set)
                for parent, children in hierarchy.items():
                    for child in children:
                        sub_num_hiers[label_to_pos[parent]].add(label_to_pos[child])

                sub_reversed_hiers = {}
                for parent, children in sub_num_hiers.items():
                    for child in children:
                        sub_reversed_hiers[child] = parent


                #sub_num_hiers {pos_to_idx[parent]: [pos_to_idx[child] for child in children] for parent, children in sub_num_hiers.items()}

          #      print('---------------------------------')
         #       print(sub_input_ids.shape)
         #       print(sub_input_ids)
          #      print(sub_attention_mask.shape)
          #      print(sub_attention_mask[0])
          #      print(sub_position_ids.shape)
           #     print(sub_position_ids)
          #      print(sub_init_label_emb.shape)
                #print(sub_init_label_emb[0])
          #      print('sub_num_hiers:')
          #      print(sub_num_hiers)
          #      print('sub_reversed_hiers:')
         #      print(sub_reversed_hiers)
          #      print('------------------------------')
         #       print(reversed_hiers)
         #       print('------------------------------')
         #       print(r_id_hiera)

                SPLIT_input_ids.append(sub_input_ids)
                SPLIT_attention_masks.append(sub_attention_mask)
                SPLIT_position_ids.append(sub_position_ids)
                SPLIT_init_label_emb.append(sub_init_label_emb)
                SPLIT_num_hiers.append(sub_num_hiers)
                SPLIT_reversed_hiers.append(sub_reversed_hiers)
                SPLIT_pos_to_idx.append(pos_to_idx)
                break
            ### CUSTOM END

#            init_label_emb = training_cpt(args, tokenizer, input_ids, attention_mask,
#                                            position_ids, init_label_emb, num_hiers, reversed_hiers).detach().cpu()
            for i in range(len(SPLIT_init_label_emb)):
                SPLIT_init_label_emb[i] = training_cpt(args, tokenizer, SPLIT_input_ids[i], SPLIT_attention_masks[i],
                                            SPLIT_position_ids[i], SPLIT_init_label_emb[i], SPLIT_num_hiers[i], SPLIT_reversed_hiers[i], SPLIT_pos_to_idx[i]).detach().cpu()
                #SPLIT_init_label_emb[i] = training_cpt(args, tokenizer, SPLIT_input_ids[i], SPLIT_attention_masks[i],
                #                            SPLIT_position_ids[i], SPLIT_init_label_emb[i], SPLIT_num_hiers[i], reversed_hiers).detach().cpu()

            label_embeddings = torch.zeros(len(label_tokens), config.hidden_size)
            for i in range(len(SPLIT_init_label_emb)):
                ids = SPLIT_ids[i]
                sub_init_label_emb = SPLIT_init_label_emb[i]
                for idx, emb in enumerate(sub_init_label_emb):
                    label_embeddings[ids[idx]] += emb
                #label_embeddings[ids] += sub_init_label_emb### CURRENT

            # for compare with same seed
            torch.set_rng_state(rng_state)
        elif args.random_label_init:
            rng_state = torch.get_rng_state()
            init_label_emb = torch.nn.Embedding(len(label_tokens), config.hidden_size).weight.data
            torch.set_rng_state(rng_state)

        model.bert.embeddings.word_embeddings.weight.data = torch.cat([model.bert.embeddings.word_embeddings.weight.data, init_label_emb], dim=0)
        model.bert.embeddings.word_embeddings.num_embeddings += len(label_tokens)
        model.cls.predictions.bias.data = torch.cat([model.cls.predictions.bias.data, torch.zeros(len(label_tokens))],
                                                        dim=0)
        vs = config.vocab_size
        config.vocab_size = config.vocab_size + len(label_tokens)
        if args.softmax_label_only:
            model.label_start_index = label_tokens_start_index
    else:
        vs = config.vocab_size

    if args.soft_label:
        model.soft_label = True
        model.mask_token_id = tokenizer.mask_token_id
        model.sep_token_id = tokenizer.sep_token_id
        model.vs = vs

    return model, tokenizer, vs
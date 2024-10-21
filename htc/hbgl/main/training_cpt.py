from collections import defaultdict

import torch

from transformers import AdamW

def training_cpt(args, tokenizer, input_ids, attention_mask,  position_ids, _init_label_emb, num_hiers, reversed_hiers, pos_to_idx):
    from transformers import BertForMaskedLM
    from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
    label_nums = input_ids.shape[0] - 2

    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    model = model.train()
    model.cuda()

    init_label_emb = _init_label_emb.float().cuda().requires_grad_()
    torch.save(init_label_emb.cpu(), 'before.pt')

    optimizer_grouped_parameters = [
        {'params': [init_label_emb, ], 'weight_decay': 0.0}
    ]
    cpt_optimizer = AdamW(optimizer_grouped_parameters, lr=args.label_cpt_lr, eps=args.adam_epsilon)

    mask_ratio = 0.15
    bs = args.label_cpt_bsz
    b_input_ids = input_ids.unsqueeze(0).repeat(bs, 1).cuda().long()
    position_ids = position_ids.unsqueeze(0).repeat(bs, 1).cuda().long()

    if args.label_cpt_decodewithpos:
        position_ids[:, 1:-1] += args.max_source_seq_length - 1
        position_ids[:, -1] = args.max_source_seq_length + args.max_target_seq_length - 1
    attention_mask = attention_mask.unsqueeze(0).repeat(bs, 1, 1).cuda().long()
    for step in range(args.label_cpt_steps):
        if args.label_cpt_not_incr_mask_ratio:
            c_mask_ratio = mask_ratio
        else:
            c_mask_ratio = mask_ratio + (step / args.label_cpt_steps) * 0.3
        inputs_embeds = torch.cat([model.bert.embeddings.word_embeddings.weight[tokenizer.cls_token_id].unsqueeze(0),
                                   init_label_emb,
                                   model.bert.embeddings.word_embeddings.weight[tokenizer.sep_token_id].unsqueeze(0),])
        inputs_embeds = inputs_embeds.unsqueeze(0).repeat(bs, 1, 1).cuda()
        mask_tokens = ~torch.bernoulli(torch.ones_like(b_input_ids) * (1 - c_mask_ratio)).bool()
        labels = torch.ones_like(b_input_ids).long() * -100
        mask_tokens[:, 0] = 0
        mask_tokens[:, -1] = 0
        labels[mask_tokens] = b_input_ids[mask_tokens] - model.bert.embeddings.word_embeddings.num_embeddings
        inputs_embeds[mask_tokens] = model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]
        outputs = model.bert(
            None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        hidden_states = model.cls.predictions.transform(sequence_output)
        prediction_scores = hidden_states @ init_label_emb.T

        if args.label_cpt_use_bce:
            loss_fct = BCEWithLogitsLoss()  # -100 index = padding token
            with torch.no_grad():
                bce_labels = torch.zeros_like(prediction_scores)
                _bce_labels = []
                for b in range(bs):
                    l = labels[b][mask_tokens[b]].tolist()
                    bce_l = bce_labels[b][mask_tokens[b]]
                    c = defaultdict(list)
                    lmap = {}
                    l = [hm + 1 for hm in l]
                    for il in l:
                        if il not in num_hiers:
                            # last labels
                            p = reversed_hiers[il]
                            c[p].append(il)
                            lmap[il] = p
                    for i, il in enumerate(l):
                        if il not in lmap:
                            il = pos_to_idx[il] - 1
                            bce_l[i][il] = 1
                        else:
                            for j in c[lmap[il]]:
                                # eigen
                                j = pos_to_idx[j] - 1
                                bce_l[i][j] = 1
                    _bce_labels.append(bce_l)
                bce_labels = torch.cat(_bce_labels, dim=0)
                print(bce_labels.sum())
            masked_lm_loss = loss_fct(prediction_scores[mask_tokens], bce_labels)
        else:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, label_nums), labels.view(-1))

        masked_lm_loss.backward()
        cpt_optimizer.step()
        model.zero_grad()
        init_label_emb.grad = None
        print(f'step {step}', masked_lm_loss.item())
    torch.save(init_label_emb.cpu(), 'after.pt')
    return init_label_emb
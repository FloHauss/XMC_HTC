import torch

from transformers import AdamW

def prepare_for_training(args, model, checkpoint_state_dict, amp):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

        # then remove optimizer state to make amp happy
        # https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
        if amp:
            optimizer.state = {}

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if checkpoint_state_dict:
            amp.load_state_dict(checkpoint_state_dict['amp'])

            # Black Tech from https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
            # forward, backward, optimizer step, zero_grad
            random_input = {'source_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'target_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'label_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'pseudo_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'num_source_tokens': torch.zeros(size=(2,), device=args.device, dtype=torch.long),
                            'num_target_tokens': torch.zeros(size=(2,), device=args.device, dtype=torch.long)}
            loss = model(**random_input)
            print("Loss = %f" % loss.cpu().item())
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            model.zero_grad()

            # then load optimizer state_dict again (this time without removing optimizer.state)
            optimizer.load_state_dict(checkpoint_state_dict['optimizer'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer
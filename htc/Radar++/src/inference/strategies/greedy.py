"""Greedy Search with optional Focal Temperature Scaling (FTS)"""
import torch

import hyperparameter

def greedy_search(config, model, input_ids, attention_mask):
    """Simple greedy search selecting the token with the highest score. """
    model = model.module # Access through DDP wrapping
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Initialize sequence with end tokens (simplifies early stopping)
    seq = torch.full(
        (batch_size, config.max_seq_len),
        config.end_idx,
        dtype=torch.long,
        device=device
    )
    seq[:, 0] = config.start_idx

    # Encode input once
    encoder_output = model.encode(input_ids, attention_mask)
    encoder_padding_mask = attention_mask == 0

    # Generate tokens sequentially
    for t in range(1, config.max_seq_len):
        # Get logits for cuttent step
        logits = model.generate(
            seq[:, :t],
            encoder_output,
            encoder_padding_mask
        )

        # Apply conditional calibaratin
        if getattr(config, 'hyperparameter_tuning', False):
            probs = hyperparameter.multi_focal_link(logits / config.temperature, config.gamma)
        else:
            probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Select tokens with highest probability
        _, predicted = torch.max(probs, dim=-1)

        # Do not extend finished sequences
        finished_mask = seq[:, t-1] == config.end_idx
        predicted[finished_mask] = config.end_idx

        seq[:, t] = predicted

        # Early stopping if all sequences finished
        if predicted.eq(config.end_idx).all():
            break

    return seq

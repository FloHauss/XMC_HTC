"""First beam search approach"""
import torch


def beam_search_simple(config, model, input_ids, attention_mask, beam_size: int = 5):
    """Simple beam search. No constraints regarding token selection."""
    model = model.module
    device = input_ids.device
    batch_size = input_ids.shape[0]
    max_seq_len = config.max_seq_len
    start_idx = config.start_idx
    end_idx = config.end_idx

    encoder_output = model.encode(input_ids, attention_mask)
    encoder_padding_mask = ~(attention_mask.to(torch.bool)).to(device)

    sequences = torch.full(
        (batch_size, beam_size, max_seq_len),
        end_idx,
        dtype=torch.long,
        device=device
    )
    sequences[:, :, 0] = start_idx

    log_probs = torch.zeros(batch_size, beam_size, device=device)

    finished = torch.zeros(batch_size, beam_size, device=device)

    for t in range(1, max_seq_len):
        all_logits = []

        for batch_idx in range(batch_size):
            batch_logits = []
            for beam_idx in range(beam_size):
                if finished[batch_idx, beam_idx]:
                    dummy_logits = torch.full(
                        (config.vocab_size,),
                        float('-inf'),
                        device=device
                    )
                    dummy_logits[end_idx] = 0
                    batch_logits.append(dummy_logits)
                else:
                    seq = sequences[batch_idx, beam_idx, :t].unsqueeze(0)
                    logits = model.generate(
                        seq,
                        encoder_output[batch_idx:batch_idx+1],
                        encoder_padding_mask[batch_idx:batch_idx+1]
                    )[0]
                    batch_logits.append(logits)
            all_logits.append(torch.stack(batch_logits))

        all_logits = torch.stack(all_logits)

        log_probs_t = torch.nn.functional.log_softmax(all_logits, dim=1)

        candidate_scores = log_probs.unsqueeze(-1) + log_probs_t

        candidate_scores_flat = candidate_scores.flatten(start_dim=1)

        top_scores, top_indices = torch.topk(candidate_scores_flat, beam_size, dim=1)

        vocab_size = all_logits.shape[-1]
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        new_sequences = torch.full_like(sequences, end_idx)
        new_log_probs = torch.zeros_like(log_probs)
        new_finished = torch.zeros_like(finished)

        for batch_idx in range(batch_size):
            for new_beam_idx in range(beam_size):
                old_beam_idx = beam_indices[batch_idx, new_beam_idx]
                token_idx = token_indices[batch_idx, new_beam_idx]

                new_sequences[batch_idx, new_beam_idx] = sequences[batch_idx, old_beam_idx].clone()

                new_sequences[batch_idx, new_beam_idx, t] = token_idx

                new_log_probs[batch_idx, new_beam_idx] = top_scores[batch_idx, new_beam_idx]

                new_finished[batch_idx, new_beam_idx] = \
                    (token_idx == end_idx) or finished[batch_idx, old_beam_idx]

        sequences = new_sequences
        log_probs = new_log_probs
        finished = new_finished

        if finished.all():
            break

    best_beam_indices = log_probs.argmax(dim=1)
    result = sequences[torch.arange(batch_size), best_beam_indices]

    return result

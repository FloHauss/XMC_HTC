"""
Constrained Beam Search with Level-based Constraints.
Using beam search parameters proposed by this paper: 
    Google's Neural Machine Translation System: 
    Bridging the Gap between Human and Machine Translation.
"""
from dataclasses import dataclass
from typing import List, Optional, Set

import torch

import hyperparameter


@dataclass
class BeamCandidate:
    """Represents a single beam candidate during search."""
    sequence: torch.Tensor
    log_prob: float
    length: int
    finished: bool
    current_level: int
    parent_beam_idx: Optional[int] = None
    token_history: Optional[List[int]] = None

    def __post__init__(self):
        if self.token_history is None:
            self.token_history = []

    def score(self, length_penalty: float = 0.6) -> float:
        """Calculate normalized score with length penalty."""
        if self.length == 0:
            return float('-inf')

        length_norm = ((5 + self.length) / 6) ** length_penalty
        return self.log_prob / length_norm

    def __lt__(self, other):
        return self.score() > other.score()

    def __eq__(self, other):
        return torch.equal(self.sequence, other.sequence) and self.finished == other.finished


class LevelConstraints:
    """Manages hierarchical level constraints for beam search."""

    def __init__(self, level_token: int, label_levels: List[List[int]],
                 start_idx: int, end_idx: int):
        self.level_token = level_token
        self.label_levels = label_levels
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_levels = len(label_levels)

        # Pre-compute sets for efficient lookup
        self.level_label_sets = [set(labels) for labels in label_levels]
        self.special_tokens = {start_idx, end_idx, level_token}

    def get_allowed_tokens(self, current_level: int, can_advance_level: bool = True) -> Set[int]:
        """Get tokens allowed at the current level."""
        allowed = {self.end_idx}  # Always allow ending

        if 0 <= current_level < self.num_levels:
            allowed.update(self.level_label_sets[current_level])

            # Allow advancing to next level if not at the last level
            if can_advance_level and current_level < self.num_levels:
                allowed.add(self.level_token)

        return allowed

    def update_level_from_token(self, current_level: int, token: int) -> int:
        """Update level based on selected token."""
        return current_level + 1 if token == self.level_token else current_level

    def is_valid_transition(self, current_level: int, token: int) -> bool:
        """Check if token transition is valid at current level."""
        allowed = self.get_allowed_tokens(current_level)
        return token in allowed


def constrained_beam_search(config, model, input_ids, attention_mask,
                            level_constraints: LevelConstraints,
                            beam_size: int = 5,
                            length_penalty: float = 0.6,
                            diversity_penalty: float = 0.0,
                            max_backtrack_steps: int = 3):
    """Perform constrained beam search with level-based constraints."""
    model = model.module  # Urwrap DDP
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Encode input once
    encoder_output = model.encode(input_ids, attention_mask)
    encoder_padding_mask = attention_mask == 0

    # Initialize beams for each batch
    all_beams = []
    completed_sequences = [[] for _ in range(batch_size)]

    for batch_idx in range(batch_size):
        initial_candidate = _create_initial_candidate(config, device)
        all_beams.append([initial_candidate])

    # Generate sequences step by step
    for t in range(1, config.max_seq_len):
        all_finished = True

        for batch_idx in range(batch_size):
            if not all_beams[batch_idx]:
                continue

            all_finished = False

            # Process current batch
            _process_batch_step(
                batch_idx, t, all_beams, completed_sequences,
                model, encoder_output, encoder_padding_mask,
                level_constraints, config, beam_size,
                length_penalty, diversity_penalty, max_backtrack_steps
            )

        if all_finished:
            break

    # Select final_sequences
    return _select_final_sequences(all_beams, completed_sequences, config, device, length_penalty)


def _create_initial_candidate(config, device):
    """Create initial beam candidate."""
    initial_seq = torch.full((config.max_seq_len,),
                             config.end_idx, dtype=torch.long, device=device)
    initial_seq[0] = config.start_idx

    return BeamCandidate(
        sequence=initial_seq.clone(),
        log_prob=0.0,
        length=1,
        finished=False,
        current_level=0,
        token_history=[config.start_idx]
    )

# The function parameters are a crime against humanity
def _process_batch_step(batch_idx, t, all_beams, completed_sequences,
                        model, encoder_output, encoder_padding_mask,
                        level_constraints, config, beam_size,
                        length_penalty, diversity_penalty, max_backtrack_steps):
    """Process one step of beam search for a single batch."""
    beams = all_beams[batch_idx]
    new_candidates = []

    # Generate candidates from each beam
    for beam_idx, beam in enumerate(beams):
        if beam.finished:
            continue

        candidates = _generate_beam_candidates(
            beam, beam_idx, t, model, encoder_output, encoder_padding_mask,
            batch_idx, level_constraints, config, beam_size,
            diversity_penalty, beams
        )
        new_candidates.extend(candidates)

    # Add finished beams
    new_candidates.extend(beam for beam in beams if beam.finished)

    # Select best candidates
    new_candidates.sort(key=lambda x: x.score(length_penalty), reverse=True)
    selected_beams = new_candidates[:beam_size]

    # Apply backtracking if needed
    if t > max_backtrack_steps:
        selected_beams = _apply_backtracking(
            selected_beams, completed_sequences[batch_idx],
            level_constraints, beam_size, t
        )

    # Update completed sequences
    for beam in selected_beams:
        if beam.finished and beam not in completed_sequences[batch_idx]:
            completed_sequences[batch_idx].append(beam)

    # Upate active beams
    all_beams[batch_idx] = [
        beam for beam in selected_beams if not beam.finished]

    # Stop if enough completed sequences
    if len(completed_sequences[batch_idx]) >= beam_size and not all_beams[batch_idx]:
        all_beams[batch_idx] = []


def _generate_beam_candidates(beam, beam_idx, t, model, encoder_output, encoder_padding_mask,
                              batch_idx, level_constraints, config, beam_size,
                              diversity_penalty, beams):
    """Generate candidate continuations for a single beam."""
    # Get model predcitions
    current_seq = beam.sequence[:t].unsqueeze(0)
    logits = model.generate(
        current_seq,
        encoder_output[batch_idx:batch_idx+1],
        encoder_padding_mask[batch_idx:batch_idx+1]
    )

    # Apply probability computation strategy
    if getattr(config, 'hyperparameter_tuning', False):
        log_probs = hyperparameter.multi_focal_link(logits / config.temperature, config.gamma)
    else:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.squeeze()

    constrained_log_probs = _apply_level_constraints(
        log_probs, level_constraints, beam.current_level
    )

    # Get top candidates
    allowed_tokens = level_constraints.get_allowed_tokens(beam.current_level)
    top_log_probs, top_indices = torch.topk(
        constrained_log_probs, min(beam_size, len(allowed_tokens))
    )

    # Create new candidates
    candidates = []
    for log_prob, token_idx in zip(top_log_probs, top_indices):
        token_idx = token_idx.item()
        log_prob = log_prob.item()

        if not level_constraints.is_valid_transition(beam.current_level, token_idx):
            continue

        candidate = _create_beam_candidate(
            beam, token_idx, log_prob, t, level_constraints,
            beam_idx, diversity_penalty, beams, config
        )
        candidates.append(candidate)

    return candidates


def _apply_level_constraints(log_probs, level_constraints, current_level):
    """Apply level-based constraint to log probabilities."""
    allowed_tokens = level_constraints.get_allowed_tokens(current_level)

    constraint_mask = torch.full_like(log_probs, float('-inf'))
    for token_idx in allowed_tokens:
        if token_idx < log_probs.shape[0]:
            constraint_mask[token_idx] = 0.0

    return log_probs + constraint_mask


def _create_beam_candidate(beam, token_idx, log_prob, t, level_constraints,
                           beam_idx, diversity_penalty, beams, config):
    """Create a new beam candidate."""
    new_seq = beam.sequence.clone()
    new_seq[t] = token_idx

    new_log_prob = beam.log_prob + log_prob
    new_length = beam.length + 1 if token_idx != config.end_idx else beam.length
    is_finished = token_idx == config.end_idx
    new_level = level_constraints.update_level_from_token(
        beam.current_level, token_idx)

    # Apply diversity penalty
    if diversity_penalty > 0:
        token_count = sum(1 for b in beams if token_idx in b.token_history)
        new_log_prob -= diversity_penalty * token_count

    return BeamCandidate(
        sequence=new_seq,
        log_prob=new_log_prob,
        length=new_length,
        finished=is_finished,
        current_level=new_level,
        parent_beam_idx=beam_idx,
        token_history=beam.token_history + [token_idx]
    )


def _apply_backtracking(selected_beams, completed_sequences, level_constraints,
                        beam_size, t):
    """Apply backtracking strategy when no active beams remain."""
    if len(selected_beams) == 0 or t <= 3:  # Backtracking limited to 3 steps. Magic Number for now.
        return selected_beams

    has_active_beams = any(not beam.finished for beam in selected_beams)

    if not has_active_beams and completed_sequences:
        backtrack_candidates = []

        # Create backtrack candidates from recent completed sequences
        for seq in completed_sequences[-2:]:
            if seq.length < t - 1:
                backtrack_candidate = BeamCandidate(
                    sequence=seq.sequence.clone(),
                    log_prob=seq.log_prob * 0.9,  # Penalty for backtracking
                    length=seq.length,
                    finished=False,
                    current_level=min(seq.current_level + 1,
                                      level_constraints.num_levels - 1),
                    token_history=seq.token_history.copy()
                )
                backtrack_candidates.append(backtrack_candidate)

        if backtrack_candidates:
            selected_beams.extend(backtrack_candidates[:beam_size//3])
            selected_beams = selected_beams[:beam_size]

    return selected_beams


def _select_final_sequences(all_beams, completed_sequences, config, device, length_penalty):
    """Select final sequences from beams and completes sequences."""
    final_sequences = []

    for batch_idx, _ in enumerate(all_beams):
        all_candidates = completed_sequences[batch_idx] + all_beams[batch_idx]

        if all_candidates:
            best_candidate = max(
                all_candidates, key=lambda x: x.score(length_penalty))
            final_sequences.append(best_candidate.sequence)
        else:
            # Create empty sequence as fallback
            empty_seq = torch.full(
                (config.max_seq_len,),
                config.end_idx,
                dtype=torch.long,
                device=device
            )
            empty_seq[0] = config.start_idx
            final_sequences.append(empty_seq)

    return torch.stack(final_sequences)

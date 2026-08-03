"""Dynamically manages inference strategies"""
from .greedy import greedy_search
from .threshold import threshold_search

from .beam_simple import beam_search_simple as beam_1
from .beam_constrained import constrained_beam_search as beam_2


def get_decoding_strategy(config):
    """Dynamically load inference strategy."""
    strategy = config.decoding_strategy
    if strategy == 'greedy':
        return greedy_search
    if strategy == 'beam':
        return beam_2
    if strategy == 'threshold':
        return threshold_search
    raise ValueError(f"Unknown strategy: {strategy}")

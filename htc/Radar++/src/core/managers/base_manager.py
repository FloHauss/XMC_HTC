"""Base Manager class"""
import logging


class BaseManager:
    """Base Manager with DDP and logging."""

    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger('log')
        self.device = None

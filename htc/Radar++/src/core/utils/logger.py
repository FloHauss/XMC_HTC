"""Logging module"""
import logging
import torch


def setup_logger(name='my_logger', level=logging.INFO):
    """Manages logging inside DDP."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    class Rank0Filter(logging.Filter):
        """Filter to avoid logging outside the main process."""

        def filter(self, _):
            no_ddp = not torch.distributed.is_available()
            no_ddp_init = not torch.distributed.is_initialized()
            rank_0 = torch.distributed.get_rank() == 0
            return no_ddp or no_ddp_init or rank_0

    logger.addFilter(Rank0Filter())
    return logger

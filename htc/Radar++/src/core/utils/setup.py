"""Device setup for DDP"""
import torch


def setup_device(world_size, rank, logger):
    """Setup device within DDP context."""
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    )

    logger.info(
        'Training on %d devices. Each device sees 1/%d of the data per epoch',
        world_size, world_size
    )
    return device

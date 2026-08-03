"""Dataloader for DDP"""
import torch

import datasets


def create_distributed_dataloader(config, dataset, rank, world_size, is_training=True):
    """Dataloader for DDP processes."""
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config.shuffle if is_training else False
    )

    collate_fn = datasets.get_collate_fn(config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return dataloader

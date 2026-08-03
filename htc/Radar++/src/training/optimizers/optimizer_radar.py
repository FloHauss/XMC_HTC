"""Optimizers for RADAr++"""
import torch

def optimizers_radar(config, model):
    """Optimizers for RADAr++"""
    model = model.module
    optimizer_dense = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.lr_encoder},
        {'params':
            list(model.decoder.parameters()) +
            list(model.fc.parameters()) +
            list(model.position_embedding.parameters()),
            'lr': config.lr_decoder
         },
    ])

    optimizer_sparse = torch.optim.SparseAdam([
        {'params': model.embedding.parameters(), 'lr': config.lr_decoder, 'sparse': True},
    ])

    scheduler_dense = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_dense, mode='min', factor=0.1, patience=config.lr_patience)
    scheduler_sparse = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_sparse, mode='min', factor=0.1, patience=config.lr_patience)

    return (optimizer_dense, optimizer_sparse), (scheduler_dense, scheduler_sparse)

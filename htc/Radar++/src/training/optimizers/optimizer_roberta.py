"""Optimizers for RoBERTa baseline"""
import torch

def optimizers_roberta(config, model):
    """Optimizers for RoBERTa baseline"""
    model = model.module
    optimizer_dense = torch.optim.AdamW([
        {
            'params': list(model.encoder.parameters()) + list(model.fc.parameters()), 
            'lr': config.lr_encoder
        },
    ])
    optimizer_sparse = None

    scheduler_dense = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_dense, mode='min', factor=0.1, patience=config.lr_patience)
    scheduler_sparse = None

    return (optimizer_dense, optimizer_sparse), (scheduler_dense, scheduler_sparse)

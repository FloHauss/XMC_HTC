"""Models and loss functions"""
from .radar import RADArModel
from .roberta import RoBERTaModel

from .loss_fn import FocalLoss, BaselineLoss


def get_model(config):
    """Dynamically load model based on config."""
    variant = config.model_variant
    if variant == 'radar':
        return RADArModel(config)
    elif variant == 'roberta':
        return RoBERTaModel(config)
    else:
        raise ValueError(f'Unknown model variant: {variant}')


def get_loss_fn(config):
    """Dynamically load loss function based on config."""
    loss_fn = config.loss
    if loss_fn == 'focal':
        return FocalLoss(
            gamma=config.gamma,
            ignore_index=config.padding_idx,
            label_smoothing=config.label_smoothing
        )
    if loss_fn == 'bce':
        return BaselineLoss()

    raise ValueError(f'Unknown loss function: {loss_fn}')

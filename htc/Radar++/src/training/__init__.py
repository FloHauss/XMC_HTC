"""Training Module. Includes Training Manager and optimizers for the models."""
from .trainer import TrainingManager

from .optimizers.optimizer_radar import optimizers_radar
from .optimizers.optimizer_roberta import optimizers_roberta

def get_optimizers(config, model):
    """Dynamically load optimizers depending on the model."""
    variant = config.model_variant
    if variant == 'radar':
        return optimizers_radar(config, model)
    if variant == 'roberta':
        return optimizers_roberta(config, model)

    raise ValueError(f"Unknown model variant: {variant}")
    
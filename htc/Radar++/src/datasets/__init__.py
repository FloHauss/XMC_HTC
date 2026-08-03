"""Dynamically manage datasets and dataloaders for models."""
from .dataset_radar import RADArDataset, collate_radar
from .dataset_roberta import RoBERTaDataset, collate_roberta


def load_dataset(config, path, tokenizer, taxonomy_manager, sequence_manager, rank):
    """Dynamically build dataset for model."""
    dataset_cls = _get_dataset_class(config.model_variant)

    return dataset_cls(
        config=config,
        path=path,
        tokenizer=tokenizer,
        taxonomy_manager=taxonomy_manager,
        sequence_manager=sequence_manager,
        rank=rank
    )


def _get_dataset_class(variant):
    """Dnymically load dataset class for model."""
    if variant == 'radar':
        return RADArDataset
    if variant == 'roberta':
        return RoBERTaDataset
    raise ValueError(
        f'Unkown variant: {variant}. Only radar and roberta variant are supported.')


def get_collate_fn(config):
    """Dynamically load collate_fn for model."""
    variant = config.model_variant
    if variant == "radar":
        return collate_radar
    if variant == "roberta":
        return collate_roberta
    raise ValueError(f"Unknown model variant: {variant}")

"""Manages tuning of learning rate during training"""
import random

import torch
import torch.utils.data.distributed
from tqdm import tqdm

import datasets
import core
import evaluation


class ValidationManager(core.BaseManager):
    """Manages model validation with loss and F1 metrics."""

    def __init__(self, config, model, dataset_val, rank, world_size, device):
        super().__init__(config, rank, world_size)

        self.model = model
        self.dataset_val = dataset_val
        self.device = device
        self.use_mixed_precision = getattr(
            config, 'use_mixed_precision', False)

        self.f1_metric = evaluation.metrics.MulticlassF1()
        self.special_indices = config.special_token_ids

    def _create_validation_dataloader(self, use_sampling=True):
        """Create validation dataloader with optional sampling."""
        dataset = self.dataset_val

        # Apply sampling if requested
        if use_sampling:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            subset_size = int(len(indices) * self.config.sample_factor)
            dataset = torch.utils.data.Subset(dataset, indices[:subset_size])

        # Setup distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        sampler.set_epoch(0)

        # Create dataloader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=datasets.get_collate_fn(self.config),
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def _forward_batch(self, batch):
        """Forward pass through model with optional mixed precision."""
        model_input = self.model.module.process_batch(batch, self.device)

        if self.use_mixed_precision:
            with torch.autocast(device_type=self.device.type):
                return self.model(**model_input)
        else:
            return self.model(**model_input)

    def _validate_with_loss(self, dataloader):
        """Validate using loss metric."""
        total_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc='Validating (Loss)',
                leave=False,
                disable=self.rank != 0
            ):
                _, loss = self._forward_batch(batch)
                total_loss += loss.item()

        return torch.tensor(total_loss) / len(dataloader)

    def _validate_with_f1(self, dataloader):
        """Validate using F1 metric."""
        self.f1_metric.reset()

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc='Validating (F1)',
                leave=False,
                disable=self.rank != 0
            ):
                logits, _ = self._forward_batch(batch)
                model_input = self.model.module.process_batch(
                    batch, self.device)

                # Process predictions for each sample in batch
                for idx in range(logits.shape[0]):
                    pred_indices = torch.argmax(logits[idx], dim=-1)

                    # Convert to vocabulary IDs
                    pred_vocab_ids = self.dataset_val.sequence_manager.to_vocab_ids(
                        pred_indices).unsqueeze(0)
                    target_vocab_ids = self.dataset_val.sequence_manager.to_vocab_ids(
                        model_input['tgt_seq'][idx]
                    ).unsqueeze(0)

                    self.f1_metric.update(pred_vocab_ids, target_vocab_ids)

        return self.f1_metric.compute()[1]

    def _synchronize_metric_across_devices(self, metric_value):
        """Synchronize metric across all devices in distributed training."""
        if self.world_size > 1:
            torch.distributed.all_reduce(
                metric_value, op=torch.distributed.ReduceOp.SUM)

        return metric_value.item() / self.world_size

    def validate(self, use_sampling=True):
        """Main validation method supporting both loss and F1 metrics."""
        dataloader = self._create_validation_dataloader(use_sampling)

        # Choose validation method based on config
        if getattr(self.config, 'f1_validation', False):
            metric_value = self._validate_with_f1(dataloader)
            synchronized_metric = 1.0 - \
                self._synchronize_metric_across_devices(metric_value)
            metric_name = 'f1_error_rate'
        else:
            metric_value = self._validate_with_loss(dataloader)
            synchronized_metric = self._synchronize_metric_across_devices(
                metric_value)
            metric_name = 'loss'

        self.logger.info('Validation %s: %.4f',
                         metric_name, synchronized_metric)
        return synchronized_metric

"""Trainer for models"""
from dataclasses import dataclass
from pathlib import Path
import shutil
import torch
from tqdm import tqdm
import core
import core.managers
from training.validation_manager import ValidationManager
from core.utils.setup import setup_device
from datasets.dataloader import create_distributed_dataloader
import training


@dataclass
class TrainingState:
    """Container for training state information."""
    min_val_metric: float = float('inf')
    best_epoch: int = -1
    patience: int = 0
    frozen_encoder: bool = False


class TrainingManager(core.BaseManager):
    """Manages the training process for our methods."""

    def __init__(self, config, dataset_train, dataset_val, rank, world_size):
        super().__init__(config, rank, world_size)

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.state = TrainingState(patience=config.patience)

        self.use_mixed_precision = getattr(
            config, 'use_mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None

        self.save_dir = Path('../saved_models') / self.config.dataset
        self.save_dir.mkdir(exist_ok=True)

        self._initialize_components()

        if self.use_mixed_precision:
            self.logger.info('Mixed precision training enabled')
        self.logger.info(
            'Training Manger initialized with model_id: %s', config.model_id)

    def _initialize_components(self):
        """Initialize all training components."""
        self.device = setup_device(self.world_size, self.rank, self.logger)

        self.model_manager = core.managers.DDPModelManager(self.config, self.device, self.rank)
        self.model = self.model_manager.setup_for_training()

        self.dataloader_train = create_distributed_dataloader(
            self.config, self.dataset_train, self.rank, self.world_size, is_training=True)

        self.optimizers, self.schedulers = training.get_optimizers(
            self.config, self.model)

        self.validator = ValidationManager(
            self.config, self.model, self.dataset_val, self.rank, self.world_size, self.device)

        self.logger.info('All training components initialized successfully')

    def train(self):
        """Main training loop with early stopping."""
        self.logger.info(
            'Starting training for %d epochs with early stopping.', self.config.epochs)

        start_epoch = int(self.config.continue_training) + \
            1 if getattr(self.config, 'continue_training', False) else 0

        try:
            for epoch in range(start_epoch, self.config.epochs):
                avg_train_loss = self._train_epoch(epoch)
                val_metric = self.validator.validate()

                self.logger.info('Epoch %d: Train Loss: %.4f, Val Metric: %.4f',
                                 epoch, avg_train_loss, val_metric)

                self._update_learning_rates(val_metric)

                if self._check_early_stopping(val_metric, epoch):
                    break

            return self._finalize_training()

        except Exception as e:
            self.logger.error('Training failed with error: %s', str(e))
            raise

    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.dataloader_train,
            desc=f'Epoch {epoch + 1}/{self.config.epochs}',
            total=len(self.dataloader_train),
            disable=self.rank != 0,
        )

        self.dataloader_train.sampler.set_epoch(epoch)
        optimizer_dense, optimizer_sparse = self.optimizers

        try:
            for step, batch in enumerate(progress_bar):
                loss = self._training_step(
                    batch, optimizer_dense, optimizer_sparse, step)
                total_loss += loss

                if self.rank == 0:
                    samples_seen = (step + 1) * \
                        self.config.batch_size * self.world_size
                    progress_bar.set_postfix(
                        samples_seen=samples_seen, loss=loss)
        finally:
            progress_bar.close()

        return total_loss / len(self.dataloader_train)

    def _training_step(self, batch, optimizer_dense, optimizer_sparse, step):
        """Execute one training batch"""
        model_input = self.model.module.process_batch(batch, self.device)

        # Forward pass with optinal mixed precision
        if self.use_mixed_precision:
            with torch.autocast(device_type=self.device.type):
                _, loss = self.model(**model_input)
                loss = loss / self.config.accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            _, loss = self.model(**model_input)
            loss = loss / self.config.accumulation_steps
            loss.backward()

        # Optimization step
        if (step + 1) % self.config.accumulation_steps == 0:
            self._optimization_step(optimizer_dense, optimizer_sparse)

        return loss.item() * self.config.accumulation_steps

    def _optimization_step(self, optimizer_dense, optimizer_sparse):
        """Perform optimization step with gradient clipping."""
        # Get non-sparse parameters for gradient clipping
        non_sparse_params = [p for p in self.model.parameters()
                             if p.grad is not None and not p.grad.is_sparse]

        if self.use_mixed_precision:
            self.scaler.unscale_(optimizer_dense)
            if optimizer_sparse:
                self.scaler.unscale_(optimizer_sparse)

            torch.nn.utils.clip_grad_norm_(non_sparse_params, max_norm=1)

            self.scaler.step(optimizer_dense)
            if optimizer_sparse:
                self.scaler.step(optimizer_sparse)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(non_sparse_params, max_norm=1)
            optimizer_dense.step()
            if optimizer_sparse:
                optimizer_sparse.step()

        # Zero gradients
        optimizer_dense.zero_grad()
        if optimizer_sparse:
            optimizer_sparse.zero_grad()

    def _update_learning_rates(self, val_error):
        """Update the learning rates of the scheduler/s"""
        scheduler_dense, scheduler_sparse = self.schedulers
        scheduler_dense.step(val_error)
        if scheduler_sparse:
            scheduler_sparse.step(val_error)

    def _check_early_stopping(self, val_metric, epoch):
        """Check whether """
        if val_metric < self.state.min_val_metric:
            improvement = self.state.min_val_metric - val_metric
            self.logger.info(
                'The model improved: Reduced validation metric from %.4f to %.4f (difference %.4f)',
                self.state.min_val_metric, val_metric, improvement
            )

            self.state.best_epoch = epoch
            self.state.min_val_metric = val_metric
            self.state.patience = self.config.patience

            # Save best model
            model_path = self.save_dir / (self.config.model_id + '_current.pt')
            checkpoint_kwargs = {
                'scaler_state': self.scaler.state_dict()} if self.use_mixed_precision else {}
            self.model_manager.save_checkpoint(
                model_path, epoch, **checkpoint_kwargs)

            return False
        else:
            self.state.patience -= 1
            if self.state.patience <= 0:
                self.logger.info(
                    'No validation improvement for %d epochs. Early stopping triggered.',
                    self.config.patience
                )
                return True

            # Special handing for radar model variant
            if self.config.model_variant == 'radar' and \
                    not self.state.frozen_encoder and \
                    self.optimizers[0].param_groups[0]['lr'] < 5e-7:
                self._freeze_encoder()

            return False

    def _freeze_encoder(self):
        """Continue to learn without encoder updates. Only available for decoder models."""
        self.logger.info(
            '%s Freezing encoder - decoder continues learning alone %s',
            '*'*10, '*'*10
        )

        # Load best model
        best_model_path = self.save_dir / \
            (self.config.model_id + '_current.pt')
        if best_model_path.exists():
            self.model_manager.load_checkpoint(best_model_path, self.model)
            self.logger.info('Loaded best model from epoch %d',
                             self.state.best_epoch)

        # Freeze encoder parameters
        encoder = self.model.module.encoder if hasattr(
            self.model, 'module') else self.model.encoder
        for param in encoder.parameters():
            param.requires_grad = False

        self.state.frozen_encoder = True

    def _finalize_training(self):
        """Finalize training and save final model."""
        if self.rank != 0 or self.state.best_epoch == -1:
            return

        # Copy best model to final model
        final_model_path = self.save_dir / (self.config.model_id + '_final.pt')
        best_model_path = self.save_dir / \
            (self.config.model_id + '_current.pt')
        shutil.copy(best_model_path, final_model_path)

        self.logger.info(
            'Training completed. Best epoch: %d with validation metric: %.4f',
            self.state.best_epoch, self.state.min_val_metric)

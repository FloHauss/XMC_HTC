"""Manages loading and savin pytorch models"""
import logging
from pathlib import Path

import torch

import models


class DDPModelManager:
    """Manager for loading and saving pytorch models."""

    def __init__(self, config, device, rank):
        self.config = config
        self.device = device
        self.rank = rank
        self.logger = logging.getLogger('log')
        self.model = None

        self.save_dir = Path('../saved_models') / self.config.dataset
        self.model_id = config.model_id

    def _create_model(self, checkpoint_path=None):
        """ Create and configure the model with optional checkpoint loading."""
        model = models.get_model(self.config)

        # Apply torch.compile if supported:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(self.device)
            if major >= 7:
                self.logger.info(
                    'Using torch.compile, device capability: %s.%s', major, minor)
                model.compile(mode='default', dynamic=True)
            else:
                self.logger.info(
                    'Not using torch.compile, device capability: %s.%s', major, minor)
        else:
            self.logger.info('Not using torch.compile (CUDA not available)')

        model = model.to(self.device)

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_model_weights(model, checkpoint_path)

        return self._wrap_with_ddp(model)

    def _load_model_weights(self, model, checkpoint_path):
        """Load model weights from checkpoint."""
        self.logger.info('Loading model from checkpoint: %s', checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle both dict and raw state_dict formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

    def _wrap_with_ddp(self, model):
        """Wrap model with DistributedDataParallel."""
        ddp_kwargs = {
            'find_unused_parameters': True
        }

        if self.device.type == 'cuda':
            ddp_kwargs['device_ids'] = [self.rank]

        return torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    def setup_for_training(self):
        """Setup model for training with optional checkpoint continuation."""
        checkpoint_path = None

        if getattr(self.config, 'continue_training', False):
            checkpoint_path = self.save_dir / \
                f'{self.model_id}_{self.config.continue_training}.pt'

        self.model = self._create_model(checkpoint_path)
        return self.model

    def setup_for_inference(self, checkpoint_path):
        """Setup model for inference from checkpoint."""
        self.model = self._create_model(checkpoint_path)
        return self.model

    def save_checkpoint(self, checkpoint_path, epoch, scaler_state=None, optimizer_states=None):
        """Save model checkpoint with optional training state."""
        if self.rank != 0:  # Only save on main process
            return

        # Get model state dict (handle DDP wrapper)
        model_state_dict = (self.model.module.state_dict() if hasattr(
            self.model, 'module') else self.model.state_dict())

        # Build checkpoint dict
        checkpoint = {
            'model_state_dict': model_state_dict,
            'epoch': epoch
        }

        if scaler_state is not None:
            checkpoint['scaler_state'] = scaler_state

        if optimizer_states is not None:
            checkpoint['optimizer_states'] = optimizer_states

        torch.save(checkpoint, checkpoint_path)
        self.logger.info('Checkpoint saved at %s for epoch %d',
                         checkpoint_path, epoch)

    def load_checkpoint(self, checkpoint_path, model):
        """Load checkpoint into existing model and return checkpoint data."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle both dict and raw state_dict formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            checkpoint = {'model_state_dict': checkpoint}

        # Load into model (handle DDP wrapper)
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        self.logger.info('Model loaded from checkpoint: %s', checkpoint_path)
        return checkpoint

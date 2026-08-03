"""RoBERTa baseline"""

import torch
import transformers

import models


class RoBERTaModel(torch.nn.Module):
    """Simple RoBERTa encoder with classifier head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = transformers.AutoModel.from_pretrained(config.encoder)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.fc = torch.nn.Linear(config.hidden_dim, config.num_true_labels)
        self.criterion = models.get_loss_fn(config)

    def _get_cls_embedding(self, input_ids, attention_mask):
        """Extract CLS token embedding from encoder output."""
        encoder_output = self.encoder(
            input_ids, attention_mask).last_hidden_state
        return encoder_output[:, 0, :]  # CLS token is at position 0

    def generate(self, input_ids, attention_mask):
        """Generate predictions without dropout."""
        cls_embedding = self._get_cls_embedding(input_ids, attention_mask)
        return self.fc(cls_embedding)

    def forward(self, input_ids, attention_mask, ground_truth):
        """Forward pass with loss calculation."""
        cls_embedding = self._get_cls_embedding(input_ids, attention_mask)
        logits = self.fc(self.dropout(cls_embedding))
        loss = self.criterion(logits, ground_truth)
        return logits, loss

    def process_batch(self, batch, device):
        """Move relevant batch tensors to device."""
        forward_params = {'input_ids', 'attention_mask', 'ground_truth'}
        return {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
            if k in forward_params
        }

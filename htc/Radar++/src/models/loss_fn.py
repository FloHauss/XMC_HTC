"""Loss functions for the models"""
import torch


class BaselineLoss(torch.nn.Module):
    """Binary Cross Entropy loss for RoBERTa baseline."""

    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, logits, targets):
        """Loss forward pass in pytorch manner."""
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits,
            target=targets
        )

        return bce


class FocalLoss(torch.nn.Module):
    """Focal loss (sequences) for RADAr++."""

    def __init__(self, gamma=0.0, alpha=None, ignore_index=-100, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, gamma=None, temperature=None):
        """Loss forward pass in pytorch manner."""
        if temperature:
            logits = logits / temperature

        ce_loss = torch.nn.functional.cross_entropy(
            input=logits,
            target=targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** (gamma if gamma else self.gamma) * ce_loss

        return focal_loss.mean()

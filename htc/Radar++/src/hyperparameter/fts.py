"""
Focal Temperature Scaling. 
Method: Improving Calibration by Relating Focal Loss, Temperature Scaling, and Properness. 
Source Code: https://github.com/slavikkom/focal_temperature_scaling
"""

import torch


def _focal_derivative(p, gamma=2):
    """Calculates the focal derivative."""
    p = torch.clamp(p, min=1e-12, max=1-1e-12)
    return (1 - p)**gamma * (gamma * torch.log(p) / (1 - p) - 1 / p)


def _focal_map(q, gamma=2):
    """Focal map as described within the paper."""
    inverse_grad = 1 / _focal_derivative(q, gamma=gamma)
    p = inverse_grad / torch.sum(inverse_grad, dim=-1, keepdim=True)
    return p


def _check_overflow(tensor_x):
    """Checks overflows."""
    return (tensor_x > torch.finfo(tensor_x.dtype).min) & \
        (tensor_x < torch.finfo(tensor_x.dtype).max)


def multi_focal_link(x, a=2):
    """Calculates focal temperature scaling for multi-class logits"""
    original_shape = x.shape
    if len(x.shape) == 3:
        batch_size, num_classes, seq_len = x.shape
        x = x.permute(0, 2, 1).reshape(-1, num_classes)

    q = torch.nn.functional.softmax(x, dim=-1)
    p = _focal_map(q, gamma=a)
    nr_classes = p.shape[1]

    # Identify rows with overflow
    overflowed_rows = ~_check_overflow(p)
    overflowed_rows_mask = overflowed_rows.any(dim=1)

    # Select overflowed rows
    if overflowed_rows.any():
        overflowed_logits = x[overflowed_rows_mask]
        overflowed_max_indices = torch.argmax(overflowed_logits, dim=1)

        p[overflowed_rows_mask] = 1e-5 / nr_classes
        p[overflowed_rows_mask, overflowed_max_indices] = 1-1e-5

    if len(original_shape) == 3:
        p = p.reshape(batch_size, seq_len, num_classes).permute(0, 2, 1)

    return p

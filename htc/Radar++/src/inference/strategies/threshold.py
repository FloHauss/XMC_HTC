"""Threshold-based binary classifcation"""
import torch

def threshold_search(config, model, input_ids, attention_mask):
    """Apply threshold-based binary classification model predcitions."""
    # model.module to unwrap DDP
    logits = model.module.generate(input_ids, attention_mask)
    probs = torch.sigmoid(logits)
    predicted_label_ids = (probs >= config.threshold).int()

    return predicted_label_ids

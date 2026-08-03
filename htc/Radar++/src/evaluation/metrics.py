"""Multi-class Classification metrics"""
import collections

import numpy as np


class MulticlassF1():
    """Computes Micro and Macro F1 scores for multi-class classification problems."""

    def __init__(self):
        # Per-class statistics for Macro-F1
        self.class_stats = collections.defaultdict(
            lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        # Global statistics for Micro-F1
        self.global_tp = 0  # True Positives
        self.global_fp = 0  # False Positives
        self.global_fn = 0  # False Negatives

    def update(self, preds, target):
        """Update metrics with a batch of predictions and targets."""
        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred_indices = preds[i].tolist()
            target_indices = target[i].tolist()

            pred_set = set(pred_indices)
            target_set = set(target_indices)
            all_classes = pred_set.union(target_set)

            # Update statistics for each class present in predictions and targets
            for cls in all_classes:
                if cls in pred_set and cls in target_set:
                    # True Positive: correctly predicted
                    self.class_stats[cls]['tp'] += 1
                    self.global_tp += 1
                elif cls in pred_set:
                    # False Positive: predicted but not in target
                    self.class_stats[cls]['fp'] += 1
                    self.global_fp += 1
                else:
                    # False Negative: in target but not in predicted
                    self.class_stats[cls]['fn'] += 1
                    self.global_fn += 1

    def _calculate_f1_score(self, tp, fp, fn):
        """Calculate F1 score from confusion scores: tp, fp, fn."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0

    def reset(self):
        """Reset all accumulated statistics."""
        self.class_stats = collections.defaultdict(
            lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.global_tp = 0
        self.global_fp = 0
        self.global_fn = 0

    def compute(self):
        """Compute final F1 scores."""
        f1_micro = self._calculate_f1_score(
            self.global_tp, self.global_fp, self.global_fn)
        f1_macro, f1_scores = self._compute_macro_f1()

        return f1_micro, f1_macro, f1_scores

    def _compute_macro_f1(self):
        """Compute class-wise F1 scores and average them for F1-macro."""
        f1_scores = []
        for _, stats in self.class_stats.items():
            f1 = self._calculate_f1_score(
                stats['tp'], stats['fp'], stats['fn'])
            f1_scores.append(f1)

        f1_macro = np.mean(f1_scores) if f1_scores else 0
        return f1_macro, f1_scores


class MulticlassPrecision():
    """Computes p@k and R-precision for multi-class classification."""

    def __init__(self, k_values):
        self.k_values = sorted(k_values)
        self.precision_at_k_scores = collections.defaultdict(lambda: [])
        self.r_precision_scores = []

    def update(self, preds, target):
        """Update metrics with a batch of predictions and targets."""
        batch_size = preds.shape[0]

        for i in range(batch_size):
            pred_indices = preds[i].tolist()
            target_indices = target[i].tolist()
            target_set = set(target_indices)

            # Compute precision@k for each k value
            for k in self.k_values:
                top_k_predictions = pred_indices[:k]
                relevant_in_top_k = len(set(top_k_predictions) & target_set)
                precision_at_k = relevant_in_top_k / k
                self.precision_at_k_scores[k].append(precision_at_k)

            # Compute R-precision (precision at rank k)
            num_relevant = len(target_set)
            top_r_predictions = pred_indices[:num_relevant]
            relevant_in_top_r = len(set(top_r_predictions) & target_set)
            r_precision = relevant_in_top_r / num_relevant
            self.r_precision_scores.append(r_precision)

    def compute(self):
        """Compute final precision scores."""
        avg_precision_at_k = {
            k: np.mean(scores) for k, scores in self.precision_at_k_scores.items()
        }
        avg_r_precision = np.mean(self.r_precision_scores)

        return avg_precision_at_k, avg_r_precision

    def reset(self):
        """Reset all accumulated statistics."""
        self.precision_at_k_scores = collections.defaultdict(list)
        self.r_precision_scores = []

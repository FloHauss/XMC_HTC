"""Hyperparameter optimization manager for model calibration and threshold search."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchmetrics
import torchmetrics.classification
from tqdm import tqdm

import core
from core.utils.setup import setup_device
from datasets.dataloader import create_distributed_dataloader
import hyperparameter


class HyperparameterManager(core.BaseManager):
    """Manages hyperparameter optimization for model calibration and threshold search."""

    def __init__(self, config, dataset_val, rank, world_size):
        super().__init__(config, rank, world_size)

        self.config = config
        self.dataset_val = dataset_val

        self.checkpoint_path = Path('../saved_models') / \
            (f'{self.config.dataset}/{self.config.model_id}' + '_final.pt')

        # Initialize components
        self.model_manager = None
        self.model = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize device, model manager, and dataloader."""
        self.device = setup_device(self.world_size, self.rank, self.logger)

        self.model_manager = core.managers.DDPModelManager(
            self.config, self.device, self.rank
        )
        self.model = self.model_manager.setup_for_inference(
            self.checkpoint_path)

        self.dataloader = create_distributed_dataloader(
            self.config,
            self.dataset_val,
            self.rank,
            self.world_size,
            is_training=False
        )

    def _generate_gamma_temperature_pairs(self):
        """Generate gamma and temperature parameter pairs for calibration."""
        gammas = np.linspace(0.1, 2.0, 20)
        temperatures = []

        for gamma in gammas:
            temperature_lower = 1 / (gamma + 1)
            temperature_upper = 1 / (gamma + 1 - np.log(gamma + 1) / 2)
            temperature = (temperature_lower + temperature_upper) / 2
            temperatures.append(temperature)

        return zip(list(gammas), temperatures)

    def _calibrate(self):
        """Perform calibration to find optimal gamma and temperature parameters."""
        metrics_tracker = defaultdict(list)
        criterion = torchmetrics.classification.MulticlassCalibrationError(
            num_classes=self.config.vocab_size,
            n_bins=15,
            norm='l1'
        )

        gamma_temp_pairs = self._generate_gamma_temperature_pairs()

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                self.dataloader,
                desc='Tuning (gamma, temperature)',
                leave=False,
                disable=self.rank != 0
            ):
                model_input = self.model.module.process_batch(
                    batch, self.device)
                target = model_input['tgt_seq']

                logits, _ = self.model(**model_input)

                # Test different gamma/temperature combinations
                for gamma, temperature in gamma_temp_pairs:
                    preds = hyperparameter.multi_focal_link(logits / temperature, gamma)
                    preds = preds.permute(0, 2, 1)
                    ece = criterion(preds, target)
                    metrics_tracker[(gamma, temperature)].append(ece.cpu().item())

        return metrics_tracker

    def calibrate(self):
        """Calibrate model parameters and return optimal gamma/temperature pair."""
        metrics_tracker = self._calibrate()

        # Calculate mean calibration error for each configuration
        key_means = {config: np.mean(losses)
                     for config, losses in metrics_tracker.items()}

        # Find best configuration
        non_nan_means = {key: value for key,
                         value in key_means.items() if not np.isnan(value)}

        if non_nan_means:
            best_config = min(non_nan_means, key=non_nan_means.get)
            return best_config

        return (2.0, 1.0)

    def _calculate_f1_metrics(self, threshold_scores, thresholds):
        """Calculate F1 scores for all thresholds."""
        best_threshold, best_f1 = 0.5, float('-inf')

        for threshold in thresholds:
            tp = threshold_scores[threshold]['tp']
            fp = threshold_scores[threshold]['fp']
            fn = threshold_scores[threshold]['fn']

            # Calculate precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Calculate recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Calculate F1 score
            f1 = (2 * precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

            return best_threshold, best_f1

    def _search_threshold(self):
        """Search for optimal classification threshold."""
        self.model.eval()

        thresholds = np.arange(0.01, 1.0, 0.01)
        threshold_scores = {thresh: {'tp': 0, 'fp': 0, 'fn': 0}
                            for thresh in thresholds}

        with torch.no_grad():
            for batch in tqdm(
                self.dataloader,
                desc='Searching optimal threshold',
                leave=False,
                disable=self.rank != 0
            ):
                model_input = self.model.module.process_batch(
                    batch, self.device)
                input_ids = model_input['input_ids']
                attention_mask = model_input['attention_mask']
                target = model_input['ground_truth']

                # Get predictions
                logits = self.model.module.generate(input_ids, attention_mask)
                probs = torch.sigmoid(logits)

                # Convert to numpy
                probs_np = probs.cpu().detach().numpy()
                targets_np = target.cpu().detach().numpy()

                # Test all thresholds
                for threshold in thresholds:
                    preds = (probs_np >= threshold).astype(int)

                    # Calculate confusion matrix components
                    tp = np.sum((preds == 1) & (targets_np == 1))
                    fp = np.sum((preds == 1) & (targets_np == 0))
                    fn = np.sum((preds == 0) & (targets_np == 1))

                    threshold_scores[threshold]['tp'] += tp
                    threshold_scores[threshold]['fp'] += fp
                    threshold_scores[threshold]['fn'] += fn

        # Find best threshold
        best_threshold, _ = self._calculate_f1_metrics(
            threshold_scores, thresholds)
        return round(best_threshold, 2)

    def search_threshold(self):
        """Search for optimal classification threshold"""
        threshold = self._search_threshold()
        return threshold

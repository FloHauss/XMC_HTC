"""Manages inference and saving of results"""
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

import core
import core.managers
from core.utils.setup import setup_device
from datasets.dataloader import create_distributed_dataloader
from inference.strategies import get_decoding_strategy
from inference.strategies.beam_constrained import LevelConstraints


class TestManager(core.BaseManager):
    """Manages model inference and result aggregation for evaluation."""

    def __init__(self, config, dataset_test, rank, world_size):
        super().__init__(config, rank, world_size)

        self.dataset_test = dataset_test
        self.use_mixed_precision = getattr(
            config, 'use_midex_precision', False)

        self.model_id = config.model_id
        self.results_id = config.results_id

        # Setup paths
        self.checkpoint_path = Path(
            '../saved_models') / config.dataset / f'{self.model_id}_final.pt'
        self.results_dir = Path('../saved_results') / config.dataset
        self.results_dir.mkdir(exist_ok=True)

        self.level_constraints = None # Hm
        self._initialize_components()

        if self.use_mixed_precision:
            self.logger.info('Mixed precision inference enabled')

    def _initialize_components(self):
        """Initialize all testing components."""
        self.device = setup_device(self.world_size, self.rank, self.logger)

        # Setup model
        self.model_manager = core.managers.DDPModelManager(
            self.config, self.device, self.rank)
        self.model = self.model_manager.setup_for_inference(
            self.checkpoint_path)

        # Setup dataloader
        self.dataloader = create_distributed_dataloader(
            self.config, self.dataset_test, self.rank, self.world_size, is_training=False
        )

        # Setup decoding strategy
        self.decoding_strategy = get_decoding_strategy(self.config)

        # Setup beam constraints if needed
        if self.config.decoding_strategy == 'beam':
            self._setup_beam_constraints()

    def _setup_beam_constraints(self):
        """Setup level constraints for beam search decoding."""
        label_ids = list(range(self.config.num_labels))
        levels = self.dataset_test.taxonomy_manager.get_level(label_ids)

        tok = self.dataset_test.sequence_manager.tokenizer
        token_levels = [[tok.token_to_id(str(label_id))
                         for label_id in level] for level in levels]

        self.level_constraints = LevelConstraints(
            self.config.level_idx, token_levels, self.config.start_idx, self.config.end_idx
        )

    def _get_predictions(self, input_ids, attention_mask):
        """Get predictions using the configured decoding strategy."""
        if self.config.decoding_strategy == 'beam':
            args = (self.config, self.model, input_ids,
                    attention_mask, self.level_constraints)
        else:
            args = (self.config, self.model, input_ids, attention_mask)

        if self.use_mixed_precision:
            with torch.autocast(device_type=self.device.type):
                return self.decoding_strategy(*args)
        else:
            return self.decoding_strategy(*args)

    def _process_radar_predictions(self, predictions, targets, file_handle):
        """Process and save predictions for radar model variant."""
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            tgt = targets[i]

            # Convert to vocabulary IDs
            predicted_indices = self.dataset_test.sequence_manager.to_vocab_ids(
                pred)
            target_indices = self.dataset_test.sequence_manager.to_vocab_ids(
                tgt)

            # Remove training specific syntax
            predicted_indices = self.dataset_test.taxonomy_manager.to_eval(
                predicted_indices)
            target_indices = self.dataset_test.taxonomy_manager.to_eval(
                target_indices)

            #print(predicted_indices)
            #print(target_indices)
            #print('-'*30)
            pickle.dump((predicted_indices, target_indices), file_handle)

    def _process_roberta_predictions(self, predictions, targets, file_handle):
        """Process and save predictions for roberta model variant."""
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            preds = torch.nonzero(pred).flatten()
            tgts = targets[i]

            # print(preds)
            # print(tgts)
            # print('-'*30)

            pickle.dump((preds, tgts), file_handle)

    def _run_inference(self):
        """Run inference on the test dataset."""
        self.model.eval()

        results_path = self.results_dir / f'{self.results_id}_{self.rank}.pkl'

        # Clear previous results
        with open(results_path, 'w', encoding='utf-8') as _:
            pass

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Inference', leave=True,
                              disable=self.rank != 0):
                input_ids = batch['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)

                # Prepare targets based on model variant
                if self.config.model_variant == 'radar':
                    targets = batch['src_seq'].to(
                        self.device, non_blocking=True)
                elif self.config.model_variant == 'roberta':
                    targets = [torch.nonzero(row).flatten()
                               for row in batch['ground_truth']]
                else:
                    targets = None

                # Get predictions
                predictions = self._get_predictions(input_ids, attention_mask)

                # Process and save results
                with open(results_path, 'ab') as f:
                    if self.config.model_variant == 'radar':
                        self._process_radar_predictions(
                            predictions, targets, f)
                    elif self.config.model_variant == 'roberta':
                        self._process_roberta_predictions(
                            predictions, targets, f)

    def _merge_results(self):
        """Merge results from all ranks and clean up partial files."""
        torch.distributed.barrier()

        if self.rank != 0:
            return

        merged_results_path = self.results_dir / f'{self.results_id}_all.pkl'
        partial_files = []

        # Merge all partial results
        with open(merged_results_path, 'wb') as fo:
            for r in range(self.world_size):
                partial_path = self.results_dir / f'{self.results_id}_{r}.pkl'
                partial_files.append(partial_path)

                with open(partial_path, 'rb') as fi:
                    while True:
                        try:
                            item = pickle.load(fi)
                            pickle.dump(item, fo)
                        except EOFError:
                            break

        # Clean up partial result files
        for partial_file in partial_files:
            if partial_file.exists():
                partial_file.unlink()
                self.logger.info(
                    'Deleted partial result file: %s', partial_file)

        self.logger.info('Merged results saved to %s', merged_results_path)

    def test(self):
        """Run complete testing pipeline."""
        self._run_inference()
        self._merge_results()

"""
Entry point for RADAr usage. 3 modes for 1. training, 2. inference/testing and 3. evaluation
"""
import logging
import os
import sys
from typing import Tuple


import torch
import transformers

import core
import core.managers
import core.utils
import datasets
import evaluation
import hyperparameter
import inference
import training


class Runner:
    """Main runner class for our methods"""

    def __init__(self, config, tokenizer, taxonomy_manager, sequence_manager, rank, world_size):
        self.config = config
        self.tokenizer = tokenizer
        self.taxonomy_manager = taxonomy_manager
        self.sequence_manager = sequence_manager
        self.rank = rank
        self.world_size = world_size
        self.data_path = f'../data/{config.dataset}/'
        self.logger = logging.getLogger('log')

    def train(self):
        """Execute training pipeline."""
        self.logger.info('Starting training...')

        dataset_train = datasets.load_dataset(
            self.config,
            self.data_path + 'train.json',
            self.tokenizer,
            self.taxonomy_manager,
            self.sequence_manager,
            self.rank
        )
        dataset_val = datasets.load_dataset(
            self.config,
            self.data_path + 'val.json',
            self.tokenizer,
            self.taxonomy_manager,
            self.sequence_manager,
            self.rank
        )

        trainer = training.TrainingManager(
            self.config,
            dataset_train,
            dataset_val,
            self.rank,
            self.world_size
        )
        trainer.train()
        self.logger.info('Training completed.')

    def infer(self):
        """Execute inference pipeline"""
        self.logger.info('Starting inference...')

        dataset_test = datasets.load_dataset(
            self.config,
            self.data_path + 'test.json',
            self.tokenizer,
            self.taxonomy_manager,
            self.sequence_manager,
            self.rank
        )

        tester = inference.TestManager(
            self.config,
            dataset_test,
            self.rank,
            self.world_size
        )
        tester.test()
        self.logger.info('Inference completed.')

    def eval(self):
        """Execute evaluation pipeline."""
        self.logger.info('Starting evaluation...')
        if self.rank == 0:
            evaluater = evaluation.EvaluationManager(
                self.config,
                self.rank,
                self.world_size,
                self.taxonomy_manager
            )
            evaluater.eval()
        self.logger.info('Evaluation completed.')

    def hyper(self):
        """Execute hyperparameter tuning/calibration"""
        if not getattr(self.config, 'hyperparameter_tuning', False):
            return

        self.logger.info('Starting hyperparameter tuning...')

        dataset_val = datasets.load_dataset(
            self.config,
            self.data_path + 'val.json',
            self.tokenizer,
            self.taxonomy_manager,
            self.sequence_manager,
            self.rank
        )

        if self.rank == 0:
            hyper = hyperparameter.HyperparameterManager(
                self.config,
                dataset_val,
                self.rank,
                self.world_size
            )

            if self.config.model_variant == 'radar':
                gamma, temperature = hyper.calibrate()
                self.config.gamma = gamma
                self.config.temperature = temperature
                self.logger.info('Calibrated gamma: %.2f, T: %.2f', gamma, temperature)
            elif self.config.model_variant == 'roberta':
                threshold = hyper.search_threshold()
                self.config.threshold = threshold
                self.logger.info('Calibrated threshold: %d', threshold)

        self.logger.info('Hyperparameter tuning completed')

    def run_pipeline(self, mode: str):
        """Run the specified pipeline mode."""
        pipeline_map = {
            'train': self.train,
            'test': lambda: (self.infer(), self.eval()),
            'infer': self.infer,
            'eval': self.eval,
            'hyper': self.hyper,
            'all': lambda: (self.train(), self.hyper(), self.infer(), self.eval()),
            'complete': lambda: (self.train(), self.hyper(), self.infer(), self.eval()),
            'after': lambda: (self.hyper(), self.infer(), self.eval())
        }

        if mode not in pipeline_map:
            raise ValueError(
                f'Unknown mode: {mode}. Available modes: {list(pipeline_map.keys())}')

        self.logger.info('Running pipeline mode: %s', mode)
        pipeline_map[mode]()


def setup_distributed_processing() -> Tuple[int, int]:
    """Setup distributed processing via DDP through env."""
    env_rank = os.getenv('RANK')
    env_world_size = os.getenv('WORLD_SIZE')

    rank = int(env_rank) if env_rank is not None else 0
    world_size = int(env_world_size) if env_world_size is not None else 1

    torch.distributed.init_process_group(
        backend='gloo',  # nccl is not compatible with sparse components
        init_method='env://',
    )

    return rank, world_size


def setup_logging(rank: int, world_size: int, env_rank: str, env_world_size: str):
    """Setup logging configuration."""
    core.utils.setup_logger('log')
    logger = logging.getLogger('log')

    logger.info('Rank: %d (%s)', rank,
                'default' if env_rank is None else 'from env')
    logger.info('World Size: %d (%s)', world_size,
                'default' if env_world_size is None else 'from env')
    logger.info('Current working directory: %s', os.getcwd())
    logger.info('Starting main')

    return logger


def parse_command_line_arguments() -> Tuple[str, str, int, str]:
    """Parse command line arguments."""
    if len(sys.argv) < 5:
        raise ValueError(
            'Usage: torchrun main.py <dataset_name> <config_name> <seed> <mode>')

    dataset_name = sys.argv[1]
    config_name = sys.argv[2]
    seed = int(sys.argv[3])
    mode = sys.argv[4]

    return dataset_name, config_name, seed, mode


def setup_configuration(dataset_name: str, config_name: str, seed: int):
    """Setup configuration file and set seed."""
    logger = logging.getLogger('log')

    config = core.utils.load_config(dataset_name, config_name)

    if 'use_mixed_precision' not in config:
        config.use_mixed_precision = bool(
            getattr(config, 'mixed_precision', False)
        )

    config.seed = seed

    core.utils.set_random_seeds(config.seed)
    logger.info('Set seed to: %d across all processes and devices', config.seed)

    return config


def setup_tokenizer_and_managers(config, rank):
    """Setup encoder tokenizer, taxonomy and sequence manager."""
    logger = logging.getLogger('log')
    data_path = f'../data/{config.dataset}/'

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.encoder)

    # Handles transfer of clear text labels to label ids
    taxonomy_manager = core.managers.TaxonomyManager(
        data_path,
        config.expansion,
        rank
    )
    num_true_labels = len(taxonomy_manager.true_mlb.classes_)
    num_train_labels = len(taxonomy_manager.train_mlb.classes_)

    logger.info('Number of true classes: %d', num_true_labels)
    logger.info('Number of training classes: %d', num_train_labels)

    # Handles transfer of label ids to token ids for the models
    sequence_manager = core.managers.SequenceManager(
        taxonomy_manager.train_mlb.classes_,
        config.tokenization_mode)
    vocab_size = sequence_manager.tokenizer.get_vocab_size()
    special_tokens = sequence_manager.tokenizer.special_token_ids

    logger.info('Vocab size: %d', vocab_size)
    logger.info('Number of special tokens: %d', len(special_tokens))

    return tokenizer, taxonomy_manager, sequence_manager


def update_config_with_computed_values(config, taxonomy_manager, sequence_manager):
    """Update config with computed values from managers."""
    logger = logging.getLogger('log')

    base_id = ''
    base_id += f'{config.model_variant}'
    if config.model_variant == 'radar':
        base_id += f'_{config.tokenization_mode}'
        base_id += '_expand' if config.expansion == 'expand' else ''
        base_id += '_reduce' if config.expansion == 'reduce' else ''
    elif config.model_variant == 'roberta':
        pass
    base_id += f'_{config.max_length}'
    base_id += f'_{config.seed}'

    model_id = base_id
    results_id = base_id

    if getattr(config, 'hyperparameter_tuning', False):
        results_id += '_hyper'
    results_id += f'_{config.decoding_strategy}'

    config.model_id = model_id
    config.results_id = results_id

    num_true_labels = len(taxonomy_manager.true_mlb.classes_)
    num_train_labels = len(taxonomy_manager.train_mlb.classes_)
    vocab_size = sequence_manager.tokenizer.get_vocab_size()

    # Update config with computed values
    config.num_labels = num_train_labels
    config.num_true_labels = num_true_labels
    config.num_train_labels = num_train_labels
    config.vocab_size = vocab_size
    config.special_token_ids = sequence_manager.tokenizer.special_token_ids
    config.start_idx = sequence_manager.tokenizer.token_to_id('<s>')
    config.end_idx = sequence_manager.tokenizer.token_to_id('</s>')
    config.unk_idx = sequence_manager.tokenizer.token_to_id('<unk>')
    config.padding_idx = sequence_manager.tokenizer.token_to_id('<pad>')
    config.level_idx = sequence_manager.tokenizer.token_to_id('<lvl>')

    # Calculate max sequence length if not set
    if 'max_seq_len' not in config:
        max_seq_len = 2  # <s> + </s>
        max_seq_len += taxonomy_manager.max_depth + 1  # <lvl> + <lvl> for last level
        max_seq_len += taxonomy_manager.max_labels * \
            (2 if config.tokenization_mode == 'xml' else 1)
        config.max_seq_len = max_seq_len

    logger.info('Max sequence length: %d', config.max_seq_len)


def main():
    """Main entry point for methods."""
    try:
        if torch.cuda.is_available():
            # Get the current device properties of the first detected GPU
            major, minor = torch.cuda.get_device_capability(0)

            # Ampere (8.x) or Hopper (9.x) and later architectures support TF32
            if major >= 8:
                torch.set_float32_matmul_precision('high')
                print(f"CUDA device detected (Compute Capability: {major}.{minor}). "
                      f"Setting float32_matmul_precision to 'high' for TF32.")
            else:
                print(f"CUDA device detected (Compute Capability: {major}.{minor}). "
                      f"TF32 not supported or enabled by default on this architecture.")
        else:
            print("No CUDA device detected. TF32 setting skipped.")

        rank, world_size = setup_distributed_processing()

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        logger = setup_logging(rank, world_size, os.getenv(
            'RANK'), os.getenv('WORLD_SIZE'))

        dataset_name, config_name, seed, mode = parse_command_line_arguments()

        config = setup_configuration(dataset_name, config_name, seed)
        if not getattr(config, 'hyperparameter_tuning', False):
            config.hyperparameter_tuning = False
        if not getattr(config, 'use_mixed_precision', False):
            config.use_mixed_precision = False
        logger.info(
            'Run Arguments:\n'
            '* Dataset: %s\n'
            '* Model: %s\n'
            '* Tokenization Mode: %s\n'
            '* Expansion: %s\n'
            '* Hyperparameter Tuning: %s\n'
            '* Using Mixed Precision: %s\n'
            '* Encoder Max Sequence Length: %s\n'
            '* Batch Size: %s',
            config.dataset, config.model_variant, config.tokenization_mode,
            config.expansion, config.hyperparameter_tuning,
            config.use_mixed_precision, config.max_length,
            config.batch_size
        )

        tokenizer, taxonomy_manager, sequence_manager = setup_tokenizer_and_managers(
            config, rank
        )

        update_config_with_computed_values(
            config, taxonomy_manager, sequence_manager
        )

        runner = Runner(
            config,
            tokenizer,
            taxonomy_manager,
            sequence_manager,
            rank,
            world_size
        )
        runner.run_pipeline(mode)

        logger.info('Execution completed successfully.')
    except Exception as e:
        logger = logging.getLogger('log')
        logger.error('Error during execution: %s', str(e))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

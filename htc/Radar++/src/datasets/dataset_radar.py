"""This module provides utility for loading datasets to pytorch"""
import json

import torch
from tqdm import tqdm


class RADArDataset(torch.utils.data.Dataset):
    """RADAr++ pytorch dataset."""

    def __init__(self, config, path, tokenizer, taxonomy_manager, sequence_manager, rank):
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = sum(1 for _ in f)

        self.config = config
        self.path = f'../data/{config.dataset}/'

        self.tokenizer = tokenizer
        self.taxonomy_manager = taxonomy_manager
        self.sequence_manager = sequence_manager

        self.text = []
        self.labels = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=self.lines, desc='Loading JSON lines', disable=rank != 0):
                data = json.loads(line)
                # At utmost 512 tokens of the text are used, with the rest being truncated.
                # We remove many of these words beforehand to increase the loading speed.
                self.text += [data['token'][:2000].lower()]
                self.labels += [data['label']]

    def __len__(self):
        return self.lines

    def __getitem__(self, idx):
        config = self.config

        text = self.text[idx]
        labels = self.labels[idx]

        # Encoder preprocessing
        encoding = self.tokenizer(
            text,
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        levels_ids_sorted = self.taxonomy_manager.group_by_level(labels)
        seq = self.sequence_manager.decoder_sequence(levels_ids_sorted)

        sequence = torch.tensor(seq, dtype=torch.long)

        start_tensor = torch.tensor(
            [self.sequence_manager.tokenizer.token_to_id('<s>')])
        end_tensor = torch.tensor(
            [self.sequence_manager.tokenizer.token_to_id('</s>')])
        padding_value = self.sequence_manager.tokenizer.token_to_id('<pad>')

        decoder_src = torch.cat((start_tensor, sequence))
        decoder_tgt = torch.cat((sequence, end_tensor))
        padding_len = config.max_seq_len - \
            sequence.shape[0] - 1  # -1 for <sos>/<eos>
        decoder_src = torch.nn.functional.pad(
            decoder_src, (0, padding_len), value=padding_value)
        decoder_tgt = torch.nn.functional.pad(
            decoder_tgt, (0, padding_len), value=padding_value)

        sample = (
            input_ids,
            attention_mask,
            decoder_src,
            decoder_tgt,
        )
        return sample


def collate_radar(batch: list) -> dict:
    """Custom collate function. Automatically converts sparse ground_truth to dense."""
    batch_input_ids = torch.stack([sample[0] for sample in batch], dim=0)
    batch_attention_mask = torch.stack([sample[1] for sample in batch], dim=0)
    batch_decoder_src = torch.stack([sample[2] for sample in batch], dim=0)
    batch_decoder_tgt = torch.stack([sample[3] for sample in batch], dim=0)

    collated_batch = {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'src_seq': batch_decoder_src,
        'tgt_seq': batch_decoder_tgt,
    }

    return collated_batch

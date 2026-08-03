"""RoBERTa baseline pytorch dataset + RoBERTa collate_fn"""
import json

import torch
from tqdm import tqdm


class RoBERTaDataset(torch.utils.data.Dataset):
    """RoBERTa pytorch dataset."""
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

        label_ids = self.taxonomy_manager.to_ids(labels)

        sparse_coords = torch.tensor(label_ids).unsqueeze(0)
        sparse_values = torch.ones(len(label_ids), dtype=torch.float32)
        sparse_size = (config.num_true_labels,)
        sparse_tensor = torch.sparse_coo_tensor(
            sparse_coords,
            sparse_values,
            size=sparse_size,
            dtype=torch.float32
        )
        ground_truth = sparse_tensor

        sample = (
            input_ids,
            attention_mask,
            ground_truth,
        )
        return sample


def collate_roberta(batch: list) -> dict:
    """Custom collate function. Automatically converts sparse ground_truth to dense."""
    batch_input_ids = torch.stack([sample[0] for sample in batch], dim=0)
    batch_attention_mask = torch.stack([sample[1] for sample in batch], dim=0)
    batch_ground_truth = torch.stack([sample[2] for sample in batch], dim=0)

    collated_batch = {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'ground_truth': batch_ground_truth.to_dense(),
    }

    return collated_batch

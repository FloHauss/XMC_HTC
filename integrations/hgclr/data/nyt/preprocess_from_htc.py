"""
Preprocess NYT data from cleaned HTC JSONL into the binary format
expected by the HGCLR model (tok.bin/idx, Y.bin/idx, split.pt,
bert_value_dict.pt, slot.pt).

Usage:
    python preprocess_from_htc.py --input-dir /path/to/nyt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from binarize import write_mmap_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    return parser.parse_args()


args = parse_args()
input_dir = args.input_dir.expanduser().resolve()
output_dir = args.output_dir.expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
SPLITS = {
    'train': 'nyt_train_all.json',
    'val':   'nyt_val_all.json',
    'test':  'nyt_test_all.json',
}

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# Build label dict from vocab file (Root is excluded)
with (BASE_DIR / 'nyt_label.vocab').open() as f:
    label_vocab = [l.strip() for l in f if l.strip() and l.strip() != 'Root']
label_dict = {label: i for i, label in enumerate(label_vocab)}
print(f'Labels: {len(label_dict)}')

# Build hierarchy from taxonomy
hiera = defaultdict(set)
with (BASE_DIR / 'nyt.taxonomy').open() as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        parent = parts[0]
        if parent == 'Root':
            continue
        for child in parts[1:]:
            if parent in label_dict and child in label_dict:
                hiera[label_dict[parent]].add(label_dict[child])

value_dict = {
    i: tokenizer.encode(v.split('/')[-1].lower(), add_special_tokens=False)
    for v, i in label_dict.items()
}
# Read and tokenize all splits in order (train, val, test)
all_tokens = []
all_labels = []
split_indices = {'train': [], 'val': [], 'test': []}
unknown_labels = set()
empty_label_rows = 0

for split_name, filename in SPLITS.items():
    path = input_dir / filename
    print(f'Loading {path} ...')
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            idx = len(all_tokens)
            split_indices[split_name].append(idx)

            token_ids = tokenizer.encode(
                d['token'].strip().lower(), truncation=True
            )
            all_tokens.append(token_ids)

            one_hot = [0] * len(label_dict)
            for lbl in d['label']:
                if lbl in label_dict:
                    one_hot[label_dict[lbl]] = 1
                else:
                    unknown_labels.add(lbl)
            if not any(one_hot):
                empty_label_rows += 1
            all_labels.append(one_hot)

if unknown_labels:
    preview = sorted(unknown_labels)[:10]
    raise ValueError(f"Found {len(unknown_labels)} labels missing from the vocabulary: {preview}")
if empty_label_rows:
    raise ValueError(f"Found {empty_label_rows} samples without a recognised gold label.")

torch.save(hiera, output_dir / 'slot.pt')
torch.save(value_dict, output_dir / 'bert_value_dict.pt')

print(f'Total samples: {len(all_tokens)}')
for s, idxs in split_indices.items():
    print(f'  {s}: {len(idxs)}')

torch.save(split_indices, output_dir / 'split.pt')

print('Binarizing tok ...')
write_mmap_dataset(str(output_dir / 'tok'), all_tokens)
print('Binarizing Y ...')
write_mmap_dataset(str(output_dir / 'Y'), all_labels)

print('Done.')

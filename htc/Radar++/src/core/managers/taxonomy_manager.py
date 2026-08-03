"""Manages the hierarchical structures"""
import collections
import json
import logging

import sklearn
import torch
from tqdm import tqdm


class TaxonomyManager:
    """Manages hierarchical taxonomy structure and label transformations."""

    def __init__(self, path_dataset, expansion, rank):
        self.children = {}  # parent -> set of children
        self.parents = {}  # child -> parent
        self.depth = {}  # node -> int
        self.max_depth = -1
        self.expansion = expansion  # default/expand/reduce
        self.max_labels = -1

        self.train_class_count = None
        self.true_class_count = None
        self.train_class_freq = []
        self.true_class_freq = []

        # MLB instances
        self.train_mlb = None
        self.true_mlb = None

        # Label sets
        self.train_labels = set()
        self.true_labels = set()
        self.meta_labels = set()
        self.metal_label_ids = None

        self.logger = logging.getLogger('log')
        self._load_taxonomy(path_dataset)
        self._calculate_label_properties(path_dataset, rank)

    def _load_taxonomy(self, path_dataset):
        """Load taxonomy structure from file."""
        path_taxonomy = path_dataset + \
            ('taxonomy-synthetic.txt' if self.expansion ==
             'expand' else 'taxonomy.txt')
        with open(path_taxonomy, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                parent = parts[0]
                children = parts[1:]

                if parent not in self.children:
                    self.children[parent] = set()
                    if parent not in self.depth:
                        self.depth[parent] = self.depth[self.parents[parent]
                                                        ] + 1 if parent in self.parents else -1

                for child in children:
                    self.children[parent].add(child)
                    self.parents[child] = parent
                    self.depth[child] = self.depth[parent] + 1
                    self.max_depth = max(self.max_depth, self.depth[child])

    def _calculate_label_properties(self, path, rank):
        """Calculate label frequencies and properties from dataset files."""
        self.train_class_count = collections.Counter()
        self.true_class_count = collections.Counter()

        for sub in ['train', 'val', 'test']:
            path_sub = path + f'{sub}.json'
            try:
                with open(path_sub, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)

                with open(path_sub, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, total=total_lines,
                                     desc=f'Scanning {sub} JSON lines', disable=rank != 0):
                        data = json.loads(line)
                        labels = data['label']

                        self.true_class_count.update(labels)

                        if self.expansion == 'expand':
                            labels = self._inject(labels)
                        elif self.expansion == 'reduce':
                            labels = self._reduce(labels)

                        self.max_labels = max(self.max_labels, len(labels))

                        if sub != 'test':
                            self.train_class_count.update(labels)
                        else:
                            for label in labels:
                                if label not in self.train_class_count:
                                    self.train_class_count[label] = 0
            except FileNotFoundError:
                self.logger.error('File %s not found. Break.', path_sub)

        self.train_class_freq = [cls for cls,
                                 _ in self.train_class_count.most_common()]
        self.true_class_freq = [cls for cls,
                                _ in self.true_class_count.most_common()]

        self.train_mlb = sklearn.preprocessing.MultiLabelBinarizer(
            classes=self.train_class_freq, sparse_output=True)
        self.train_mlb.fit(self.train_class_freq)

        self.true_mlb = sklearn.preprocessing.MultiLabelBinarizer(
            classes=self.true_class_freq, sparse_output=True)
        self.true_mlb.fit(self.true_class_freq)

        self.train_labels = set(self.train_mlb.classes_)
        self.true_labels = set(self.true_mlb.classes_)
        self.meta_labels = self.train_labels - self.true_labels

        for idx, label in enumerate(self.train_mlb.classes_):
            self.depth[idx] = self.depth[label]

        self.true_label_ids = torch.tensor(
            self.true_mlb.transform([self.true_labels]).indices)

    def _get_path_to_root(self, label):
        """Get the path from a label to the root node."""
        path = [label]
        current = label

        while current != 'Root' and current in self.parents:
            current = self.parents[current]
            path.append(current)

        return list(reversed(path))

    def _inject(self, labels):
        """Inject ancestor labels into the label set."""
        injected_labels = set()
        for label in labels:
            full_path = self._get_path_to_root(label)
            injected_labels.update(full_path)
        injected_labels.discard('Root')  # Root is not an actual label
        return injected_labels

    def _reduce(self, labels):
        """Remove parent labels when their children are present."""
        labels = set(labels)
        to_remove = set()

        for label in labels:
            if label in self.children:
                children = self.children[label]
                if any(child in labels for child in children):
                    to_remove.add(label)

        return labels - to_remove

    def _reconstruct(self, labels):
        """Reconstruct full hierarchy by adding all ancestors."""
        result = set(labels)

        for label in labels:
            current = label
            while current in self.parents:
                parent = self.parents[current]
                result.add(parent)
                current = parent

        return result

    def get_level(self, labels):
        """Group labels by their depth level."""
        levels = [[] for _ in range(self.max_depth + 1)]
        for label in labels:
            depth = self.depth[label]
            levels[depth].append(label)
        return levels

    def group_by_level(self, labels):
        """Group labels by level after applying expansion/reduction."""
        if self.expansion == 'expand':
            labels = self._inject(labels)
        elif self.expansion == 'reduce':
            labels = self._reduce(labels)

        label_ids = self.train_mlb.transform([labels]).indices
        levels_ids = self.get_level(label_ids)
        return [sorted(level) for level in levels_ids]

    def to_ids(self, labels):
        """Convert labels to IDs using true label binarizer"""
        return self.true_mlb.transform([labels]).indices

    def to_eval(self, labels):
        """Convert training labels to evaluation format."""
        train_label_ids = labels.tolist()
        train_labels = [self.train_mlb.classes_[id] for id in train_label_ids]

        if self.expansion == 'reduce':
            train_labels = self._reconstruct(train_labels)

        levels = self.get_level(train_labels)
        true_label_ids = []

        for level in levels:
            for label in level:
                if label in self.true_labels:
                    true_label_id = self.true_mlb.transform(
                        [[label]]).indices[0]
                    true_label_ids.append(true_label_id)

        return torch.tensor(true_label_ids, device=labels.device)

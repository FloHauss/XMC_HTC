"""For managing labels as token sequences with RADAr++"""
import itertools
import logging
import math

import tokenizers
import tokenizers.models
import tokenizers.pre_tokenizers
import torch


class SequenceManager:
    """Manages tokenization and sequence encoding for HTC."""

    def __init__(self, vocabulary, tokenization_mode):
        self.special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<lvl>']
        self.unk_token = '<unk>'
        self.tokenization_mode = tokenization_mode

        vocab_ids = list(range(len(vocabulary)))

        if tokenization_mode == 'xml':
            self.tokenizer = self._create_xml_tokenizer(vocab_ids)
        elif tokenization_mode == 'htc':
            self.tokenizer = self._create_htc_tokenizer(vocab_ids)
        else:
            raise ValueError('tokenization_mode must be "htc" or "xml"')

        self.tokenizer.special_token_ids = {
            self.tokenizer.token_to_id(token) for token in self.special_tokens}

    def _create_htc_tokenizer(self, vocabulary):
        """Create a hierarchical text classification tokenizer."""
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        vocab.update({str(label): i + len(self.special_tokens)
                     for i, label in enumerate(vocabulary)})

        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocab=vocab, unk_token=self.unk_token))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        return tokenizer

    def _create_xml_tokenizer(self, vocab_ids):
        """Create an XML-based tokenizer with base encoding."""
        max_vocab_id = max(vocab_ids) if vocab_ids else 0
        base = int(math.ceil(math.sqrt(max_vocab_id + 1)))

        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        vocab.update({f'<{i}>': i + len(self.special_tokens)
                     for i in range(base)})

        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocab=vocab, unk_token=self.unk_token))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        # Store encoding parameters
        tokenizer.base = base
        tokenizer.encoding_token_start = len(self.special_tokens)
        tokenizer.max_vocab_id = max_vocab_id

        # Add encoding/decoding methods
        tokenizer.encode_vocab_id = lambda vid: self._encode_vocab_id(
            tokenizer, vid)
        tokenizer.decode_vocab_id = lambda pair: self._decode_vocab_id(
            tokenizer, pair)
        tokenizer.encode_vocab_sequence = lambda vids: [
            tid for vid in vids for tid in tokenizer.encode_vocab_id(vid)]
        tokenizer.decode_vocab_sequence = lambda tids, skip_sepcial=True: \
            self._decode_vocab_sequence(tokenizer, tids, skip_sepcial)

        return tokenizer

    def _encode_vocab_id(self, tokenizer, vocab_id):
        """Encode a vocabulary ID into a piar of tokens."""
        if vocab_id > tokenizer.max_vocab_id:
            raise ValueError(
                f'vocab_id {vocab_id} exceeds maximum {tokenizer.max_vocab_id}')

        high, low = divmod(vocab_id, tokenizer.base)
        return [tokenizer.encoding_token_start + high, tokenizer.encoding_token_start + low]

    def _decode_vocab_id(self, tokenizer, token_pair):
        """Decode a pair of tokens back to a vocabulary ID."""
        if len(token_pair) != 2:
            raise ValueError('Expected exactly 2 tokens for decoding')

        high_idx = token_pair[0] - tokenizer.encoding_token_start
        low_idx = token_pair[1] - tokenizer.encoding_token_start

        if not (0 <= high_idx < tokenizer.base and 0 <= low_idx < tokenizer.base):
            raise ValueError('Invalid encoding tokens for decoding')

        return high_idx * tokenizer.base + low_idx

    def _decode_vocab_sequence(self, tokenizer, token_ids, skip_special_tokens=True):
        """Decode a sequence of tokens back to vocabulary IDs."""
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid >=
                         tokenizer.encoding_token_start]

        if len(token_ids) % 2 == 0:
            raise ValueError('Token sequence length must be even for decoding')

        return [tokenizer.decde_vocab_id(token_ids[i:i+2]) for i in range(0, len(token_ids), 2)]

    def decoder_sequence(self, levels):
        """Generate a decoder sequence from hierarchical levels."""
        seq = []
        for level in levels:
            if not level:
                continue

            for label_id in level:
                if self.tokenization_mode == 'htc':
                    seq.append(self.tokenizer.token_to_id(str(label_id)))
                else:  # xml
                    seq.extend(self.tokenizer.encode_vocab_id(label_id))

            seq.append(self.tokenizer.token_to_id('<lvl>'))

        return seq

    def to_vocab_ids(self, sequence):
        """Convert a token sequence to vocabulary IDs."""
        if self.tokenization_mode == 'htc':
            offset = len(self.special_tokens)
            mask = sequence >= offset
            return torch.unique(sequence[mask]) - offset

        else:  # xml
            exclude_tokens = [self.tokenizer.token_to_id(token) for token in [
                '<s>', '</s>', '<pad>']]
            level_token = self.tokenizer.token_to_id('<lvl>')

            mask = ~torch.isin(sequence, torch.tensor(
                exclude_tokens, device=sequence.device))
            filtered = sequence[mask].tolist()

            # Group by level separators and decode pairs
            groups = [list(g) for k, g in itertools.groupby(
                filtered, lambda x: x == level_token) if not k]
            sequence_ids = []

            for group in groups:
                for i in range(0, len(group) - 1, 2):
                    try:
                        sequence_ids.append(
                            self.tokenizer.decode_vocab_id(group[i:i+2]))
                    except ValueError as e:
                        logger = logging.getLogger('log')
                        logger.error('Error decoding pair %s: %s',
                                     group[i:i+2], e)

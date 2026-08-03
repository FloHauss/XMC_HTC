"""RADAr++ Model"""

import torch
import transformers

import models


class RADArModel(torch.nn.Module):
    """RADAr++ encoder-decoder model with transformer architecture."""

    def __init__(self, config):
        # super(RADArModel, self).__init__()
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = transformers.AutoModel.from_pretrained(config.encoder)

        # Embeddings
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=config.padding_idx,
            sparse=True
        )
        self.position_embedding = torch.nn.Embedding(
            config.max_seq_len,
            config.embedding_size
        )  # Currently not used. Might add it back for future configurations

        # Decoder
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            batch_first=True,  # Eases model input but requires permutation for loss calculation
            dropout=config.dropout,
            dim_feedforward=config.embedding_size * config.forward_expansion
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers)

        # Output layers
        self.fc = torch.nn.Linear(config.hidden_dim, config.vocab_size)
        self.dropout = torch.nn.Dropout(config.dropout)

        # Loss function
        self.criterion = models.get_loss_fn(config)

        # Register buffers
        self._register_masks_and_indices(config)

    def _register_masks_and_indices(self, config):
        """Registers masks and padding indice as buffers."""
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            config.max_seq_len,
            dtype=torch.bool
        )
        self.register_buffer('tgt_mask', tgt_mask)
        self.register_buffer('padding_idx', torch.tensor(config.padding_idx))

    @torch.compiler.disable
    def _sparse_embedding_forward(self, seq):
        """Forward pass through sparse embedding layer."""
        return self.embedding(seq)

    def encode(self, input_ids, attention_mask):
        """Encode input sequence using the encoder."""
        return self.encoder(input_ids, attention_mask).last_hidden_state

    def generate(self, tgt_seq, encoder_output, encoder_padding_mask):
        """Generate predictions for inference"""
        seq_len = tgt_seq.shape[1]
        tgt = self._sparse_embedding_forward(tgt_seq)
        tgt_mask = self.tgt_mask[:seq_len, :seq_len]

        # During inference, we only need memory key padding mask
        decoder_output = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=encoder_padding_mask,
            tgt_key_padding_mask=None  # Not required during inference
        )[:, -1, :]  # We append only the latest token to the sequence

        return self.fc(decoder_output)

    def forward(self, input_ids, attention_mask, src_seq, tgt_seq):
        """Forward pass with loss calculation."""
        # Encode input
        encoder_output = self.encode(input_ids, attention_mask)

        # Embed and apply dropout
        embedded = self._sparse_embedding_forward(src_seq)
        x = self.dropout(embedded)

        # Create padding masks
        encoder_padding_mask = attention_mask == 0
        tgt_key_padding_mask = src_seq == self.padding_idx

        # Decode
        decoder_output = self.decoder(
            tgt=x,
            memory=encoder_output,
            memory_key_padding_mask=encoder_padding_mask,
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Calculate loss
        logits = self.fc(decoder_output)
        loss = self.criterion(logits.permute(0, 2, 1), tgt_seq)

        return logits, loss

    def process_batch(self, batch, device):
        """Move relevant batch tensors to device."""
        forward_params = {'input_ids', 'attention_mask', 'src_seq', 'tgt_seq'}
        return {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
            if k in forward_params
        }

from __future__ import annotations

import torch
from torch import nn


class CharRNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        rnn_type = rnn_type.lower()

        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.norm(out)
        logits = self.fc(out)
        return logits, hidden

"""Bidirectional LSTM baseline used in Exp2.

``pack_padded_sequence`` requires lengths on CPU. Charge is cast to float so it
can be concatenated with the LSTM state; the released baseline stored charge as
a long tensor and the linear layer cannot consume that dtype on current PyTorch.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from pep2ccs.model.blocks import as_column


class _Embedding(nn.Module):
    def __init__(self, embedding_size: int, vocab_size: int = 32):
        super().__init__()
        self.seq_embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.seq_embedding(seq)


class _Encoder(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers,
            bias=False,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, seq: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        lengths = length.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        packed = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.LSTM(packed)
        return torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=1)


class _Regression(nn.Module):
    def __init__(self, embedding_size: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size + 1, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, final_state: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([final_state, as_column(charge)], dim=1))


class LSTMCCS(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, dropout: float, max_len: int = 64):
        super().__init__()
        self.max_len = max_len
        self.Embedding = _Embedding(embedding_size)
        self.Encoder = _Encoder(num_layers, embedding_size, dropout)
        self.Regression = _Regression(embedding_size, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.Embedding(batch["seq"])
        hidden = self.Encoder(hidden, batch["length"])
        return self.Regression(hidden, batch["charge"])

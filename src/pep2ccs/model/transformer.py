"""Transformer CCS models. Parameter names follow the released ``.pt`` files."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from pep2ccs.model.blocks import (
    CompositionPooling,
    DeepHead,
    Encoder,
    InjectedEmbedding,
    MixPooling,
    PlainDeepHead,
    ShallowHead,
    TokenEmbedding,
    as_column,
)


class TransformerCCS(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_size: int,
        num_heads: int,
        dropout: float,
        regress_dropout: float,
        max_len: int = 64,
        pooling: str = "cls_cnn",
        inject: Optional[str] = None,
        head: str = "both",
        vocab_size: int = 32,
        plain_head: bool = False,
    ):
        super().__init__()
        self.inject = inject
        self.head = head
        self.pooling_kind = pooling
        n_features = {"mz": 2, "length": 2, "joint": 3}.get(inject or "", 0)
        if inject is None:
            self.Embedding = TokenEmbedding(embedding_size, max_len, vocab_size=vocab_size)
        else:
            self.Embedding = InjectedEmbedding(embedding_size, max_len, n_features, vocab_size=vocab_size)
        self.Encoder = Encoder(num_layers, embedding_size, num_heads, dropout)
        self.Pooling = MixPooling(embedding_size, pooling)
        pooled_width = embedding_size * (2 if pooling in {"cls_max", "cls_cnn", "mean_max", "mean_cnn"} else 1)
        extra = {"both": 2, "length": 1, "mz": 1, "none": 0, "length_mz_charge": 3}[head]
        if plain_head:
            self.Regression = PlainDeepHead(pooled_width + extra, embedding_size)
        else:
            self.Regression = DeepHead(pooled_width + extra, embedding_size, regress_dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["seq"]
        mask = seq.eq(0)
        charge = batch["charge"]
        length = batch["length"]
        mz = batch["mz"]
        if self.inject is None:
            hidden = self.Embedding(seq)
        elif self.inject == "mz":
            features = torch.cat([as_column(charge), as_column(mz).sqrt()], dim=1)
            hidden = self.Embedding(seq, features)
        elif self.inject == "length":
            features = torch.cat([as_column(charge), as_column(length)], dim=1)
            hidden = self.Embedding(seq, features)
        elif self.inject == "joint":
            features = torch.cat([as_column(charge), as_column(length), as_column(mz)], dim=1)
            hidden = self.Embedding(seq, features)
        else:
            raise ValueError("unknown feature injection: {0}".format(self.inject))
        hidden = self.Encoder(hidden, mask)
        pooled = self.Pooling(hidden, mask, length)
        sqrt_mz = as_column(mz).sqrt()
        if self.head == "both":
            out = self.Regression(pooled, length, sqrt_mz)
        elif self.head == "length":
            out = self.Regression(pooled, length)
        elif self.head == "mz":
            out = self.Regression(pooled, sqrt_mz)
        elif self.head == "none":
            out = self.Regression(pooled)
        elif self.head == "length_mz_charge":
            # Same two scalars as the main model, then numeric charge.
            out = self.Regression(pooled, length, sqrt_mz, charge)
        else:
            raise ValueError("unknown regression head: {0}".format(self.head))
        return out + as_column(batch["ccs2"])


class CompositionCCS(nn.Module):
    """Exp4: composition vector and m/z enter the regression head."""

    def __init__(
        self,
        num_layers: int,
        embedding_size: int,
        num_heads: int,
        dropout: float,
        max_len: int = 64,
        composition_dim: int = 23,
    ):
        super().__init__()
        self.Embedding = TokenEmbedding(embedding_size, max_len)
        self.Encoder = Encoder(num_layers, embedding_size, num_heads, dropout)
        self.Pooling = CompositionPooling(embedding_size)
        in_features = 2 * embedding_size + composition_dim + 2
        self.Regression = ShallowHead(in_features, embedding_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["seq"]
        mask = seq.eq(0)
        hidden = self.Encoder(self.Embedding(seq), mask)
        pooled = self.Pooling(hidden, mask)
        sqrt_mz = as_column(batch["mz"]).sqrt()
        features = torch.cat(
            [batch["vector"].float(), as_column(batch["length"]), pooled, sqrt_mz],
            dim=1,
        )
        return self.Regression(features) + as_column(batch["ccs2"])

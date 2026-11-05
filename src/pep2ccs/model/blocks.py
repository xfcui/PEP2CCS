"""Layers whose parameter names match the published checkpoints."""

import torch
import torch.nn as nn


def as_column(value: torch.Tensor) -> torch.Tensor:
    """Return a float tensor of shape (batch, 1)."""
    value = value.float()
    if value.ndim == 0:
        value = value.view(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(-1)
    elif value.ndim == 2 and value.shape[-1] == 1:
        pass
    else:
        raise ValueError("expected a scalar feature, got shape {0}".format(tuple(value.shape)))
    return value


def cnn_stack(channels: int, kernel_size: int) -> nn.Sequential:
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv1d(channels, channels, kernel_size, padding=padding),
        nn.SELU(),
    )


class TokenEmbedding(nn.Module):
    def __init__(self, embedding_size: int, max_len: int, vocab_size: int = 32):
        super().__init__()
        self.max_len = max_len
        self.seq_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, _ = seq.shape
        position = torch.arange(0, self.max_len, device=seq.device).expand(batch_size, self.max_len)
        return self.seq_embedding(seq) + self.position_embedding(position)


class InjectedEmbedding(nn.Module):
    """Replace the charge-token embedding with an MLP over physical features."""

    def __init__(self, embedding_size: int, max_len: int, n_features: int, vocab_size: int = 32):
        super().__init__()
        self.max_len = max_len
        self.seq_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, seq: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch_size, _ = seq.shape
        position = torch.arange(0, self.max_len, device=seq.device).expand(batch_size, self.max_len)
        hidden = self.seq_embedding(seq)
        hidden = hidden.clone()
        hidden[:, 1, :] = self.mlp(features)
        return hidden + self.position_embedding(position)


class Encoder(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        d_ff = 4 * embedding_size
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(embedding_size, num_heads, dropout, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_size, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, embedding_size),
                )
                for _ in range(num_layers)
            ]
        )
        self.norms1 = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for index in range(self.num_layers):
            residual = seq
            seq = self.norms1[index](seq)
            seq, _ = self.attn[index](seq, seq, seq, key_padding_mask=mask)
            seq = seq + residual
            residual = seq
            seq = self.norms2[index](seq)
            seq = self.mlp[index](seq)
            seq = seq + residual
        return self.norm3(seq)


class PlainDeepHead(nn.Module):
    """Three-layer MLP without dropout. Matches checkpoints/exp2/pep2ccs.pt."""

    def __init__(self, in_features: int, embedding_size: int):
        super().__init__()
        half = embedding_size // 2
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, half),
            nn.LayerNorm(half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )

    def forward(self, pooled: torch.Tensor, *columns: torch.Tensor) -> torch.Tensor:
        if columns:
            pooled = torch.cat([pooled, *[as_column(column) for column in columns]], dim=1)
        return self.mlp(pooled)


class DeepHead(nn.Module):
    def __init__(self, in_features: int, embedding_size: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size // 2),
            nn.LayerNorm(embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, 1),
        )

    def forward(self, pooled: torch.Tensor, *columns: torch.Tensor) -> torch.Tensor:
        if columns:
            pooled = torch.cat([pooled, *[as_column(column) for column in columns]], dim=1)
        return self.mlp(pooled)


class ShallowHead(nn.Module):
    def __init__(self, in_features: int, embedding_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class MixPooling(nn.Module):
    """Pooling ablations from Exp5, plus the main model's charge-token + CNN max pool.

    ``length`` is divided on the same device as the activations. The original
    mean-pooling variants called ``length.to("cuda")``, which fails on a machine
    whose selected device is CPU or ``cuda:0`` when that call is interpreted as
    a missing device.
    """

    def __init__(self, embedding_size: int, kind: str):
        super().__init__()
        self.kind = kind
        if kind in {"cls", "cnn_max", "cls_cnn", "mean_cnn"}:
            self.local_agg = cnn_stack(embedding_size, 3)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, length: torch.Tensor = None) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1).float()
        if self.kind == "cls":
            cls = seq[:, 1, :]
            self._cnn_max(seq, valid)
            return cls
        if self.kind == "cnn_max":
            return self._cnn_max(seq, valid)
        if self.kind == "cls_cnn":
            cls = seq[:, 1, :]
            return torch.cat([cls, self._cnn_max(seq, valid)], dim=1)
        if self.kind == "cls_max":
            cls = seq[:, 1, :]
            return torch.cat([cls, self._hard_max(seq, valid)], dim=1)
        if self.kind == "max":
            return self._hard_max(seq, valid)
        if self.kind == "mean":
            return self._mean(seq, valid, length)
        if self.kind == "mean_max":
            pooled_max = self._hard_max(seq, valid)
            pooled_mean = self._mean(seq, valid, length)
            return torch.cat([pooled_mean, pooled_max], dim=1)
        if self.kind == "mean_cnn":
            pooled_mean = self._mean(seq, valid, length)
            return torch.cat([pooled_mean, self._cnn_max(seq * valid, valid)], dim=1)
        raise ValueError("unknown pooling kind: {0}".format(self.kind))

    def _cnn_max(self, seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        seq = seq * valid
        seq = seq.permute(0, 2, 1)
        valid = valid.permute(0, 2, 1)
        seq = self.local_agg(seq) * valid
        seq = seq.permute(0, 2, 1)
        pooled, _ = torch.max(seq, dim=1)
        return pooled

    @staticmethod
    def _hard_max(seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        filled = seq * valid + (1.0 - valid) * -1e9
        pooled, _ = torch.max(filled, dim=1)
        return pooled

    @staticmethod
    def _mean(seq: torch.Tensor, valid: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        if length is None:
            raise ValueError("mean pooling requires peptide length")
        denom = as_column(length).to(device=seq.device, dtype=seq.dtype)
        return (seq * valid).sum(dim=1) / denom


class CompositionPooling(nn.Module):
    """Exp4 pooling: attention pool of position 0's branch plus three CNN max-pools."""

    def __init__(self, embedding_size: int):
        super().__init__()
        self.local_agg1 = self._make_cnnblock(embedding_size, embedding_size, 3, 1, 1)
        self.local_agg2 = self._make_cnnblock(embedding_size, embedding_size, 5, 1, 1)
        self.local_agg3 = self._make_cnnblock(embedding_size, embedding_size, 7, 1, 1)
        self.global_linear = nn.Linear(embedding_size, 1)

    @staticmethod
    def _make_cnnblock(in_channels, out_channels, kernel_size, dilation, nlayers):
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation * int((kernel_size - 1) / 2),
                dilation=dilation,
            ),
            nn.SELU(),
        ]
        for _ in range(nlayers - 1):
            layers.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=dilation * int((kernel_size - 1) / 2),
                    dilation=dilation,
                )
            )
            layers.append(nn.SELU())
        return nn.Sequential(*layers)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1).float()
        seq = seq * valid
        scores = self.global_linear(seq)
        scores = torch.where(valid == 0.0, torch.tensor(float("-inf"), device=scores.device), scores)
        weights = torch.softmax(scores, dim=1)
        attended = torch.sum(weights * seq, dim=1)
        seq = seq.permute(0, 2, 1)
        valid = valid.permute(0, 2, 1)
        seq = self.local_agg1(seq) * valid
        seq = self.local_agg2(seq) * valid
        seq = self.local_agg3(seq) * valid
        seq = seq.permute(0, 2, 1)
        pooled, _ = torch.max(seq, dim=1)
        return torch.cat([attended, pooled], dim=1)

"""Build a published architecture by name."""

from typing import Dict

from pep2ccs.model.lstm import LSTMCCS
from pep2ccs.model.transformer import CompositionCCS, TransformerCCS

# pooling / feature-injection settings that reproduce each released state_dict.
_TRANSFORMERS = {
    "pep2ccs": {"pooling": "cls_cnn", "inject": None, "head": "both"},
    "exp3_mz_embed": {"pooling": "cls_cnn", "inject": "mz", "head": "length"},
    "exp3_length_embed": {"pooling": "cls_cnn", "inject": "length", "head": "mz"},
    "exp3_joint_embed": {"pooling": "cls_cnn", "inject": "joint", "head": "none"},
    "exp5_cls": {"pooling": "cls", "inject": None, "head": "both"},
    "exp5_cls_max": {"pooling": "cls_max", "inject": None, "head": "both"},
    "exp5_max": {"pooling": "max", "inject": None, "head": "both"},
    "exp5_cnn_max": {"pooling": "cnn_max", "inject": None, "head": "both"},
    "exp5_mean": {"pooling": "mean", "inject": None, "head": "both"},
    "exp5_mean_max": {"pooling": "mean_max", "inject": None, "head": "both"},
    "exp5_mean_cnn": {"pooling": "mean_cnn", "inject": None, "head": "both"},
    # checkpoints/exp2/pep2ccs.pt: vocab 50, regression input 259, no dropout in the head.
    "exp2_pep2ccs": {
        "pooling": "cls_cnn",
        "inject": None,
        "head": "length_mz_charge",
        "vocab_size": 50,
        "plain_head": True,
    },
}


def build_model(name: str, hparams: Dict, dropout: float = None, regress_dropout: float = None):
    """Create a model. Inference should pass dropout 0, matching the original scripts."""
    embedding_size = int(hparams["embedding_size"])
    num_layers = int(hparams["num_layers"])
    num_heads = int(hparams["num_heads"])
    max_len = int(hparams.get("max_len", 64))
    configured = float(hparams["dropout"])
    attn_dropout = configured if dropout is None else float(dropout)
    head_dropout = configured if regress_dropout is None else float(regress_dropout)
    if name == "lstm":
        # The released LSTM checkpoint has three layers. model.yaml's num_layers
        # is the transformer depth and does not match this file.
        return LSTMCCS(3, embedding_size, attn_dropout, max_len=max_len)
    if name == "exp4":
        return CompositionCCS(num_layers, embedding_size, num_heads, attn_dropout, max_len=max_len)
    try:
        spec = _TRANSFORMERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(list(_TRANSFORMERS) + ["lstm", "exp4"]))
        raise ValueError("unknown model {0}. Known models: {1}".format(name, known)) from exc
    return TransformerCCS(
        num_layers=num_layers,
        embedding_size=embedding_size,
        num_heads=num_heads,
        dropout=attn_dropout,
        regress_dropout=head_dropout,
        max_len=max_len,
        pooling=spec["pooling"],
        inject=spec["inject"],
        head=spec["head"],
        vocab_size=int(spec.get("vocab_size", 32)),
        plain_head=bool(spec.get("plain_head", False)),
    )


def model_names():
    return sorted(list(_TRANSFORMERS) + ["lstm", "exp4"])

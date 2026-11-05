"""Load published weights and hyper-parameters."""

import zipfile
from pathlib import Path
from typing import Dict

import torch
import yaml

from pep2ccs.paths import CHECKPOINT_DIR, CONFIG_DIR, ROOT

# Released weights are stored compressed. Each archive stays under GitHub's
# file-size limit. Members use paths relative to checkpoints/, such as exp2/lstm.pt.
ARCHIVES = (
    CHECKPOINT_DIR / "archives" / "pep2ccs.zip",
    CHECKPOINT_DIR / "archives" / "exp2.zip",
    CHECKPOINT_DIR / "archives" / "exp3.zip",
    CHECKPOINT_DIR / "archives" / "exp4.zip",
    CHECKPOINT_DIR / "archives" / "exp5-part1.zip",
    CHECKPOINT_DIR / "archives" / "exp5-part2.zip",
    CHECKPOINT_DIR / "archives" / "exp5-part3.zip",
)

DEFAULT_HPARAMS: Dict[str, float] = {
    "embedding_size": 128,
    "num_layers": 6,
    "dropout": 0.2,
    "batch_size": 512,
    "learning_rate": 0.001,
    "epochs": 300,
    "reduce_lr_step": 200,
    "eval_interval": 10,
    "early_stop": 15,
    "num_heads": 8,
    "max_len": 64,
    "split_seed": 1115,
    "valid_fraction": 0.01,
}


def load_hparams(path=None) -> Dict:
    source = CONFIG_DIR / "model.yaml" if path is None else path
    with open(source, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    hparams = dict(DEFAULT_HPARAMS)
    hparams.update(loaded or {})
    return hparams


def load_experiment(name: str) -> Dict:
    path = CONFIG_DIR / "experiments" / "{0}.yaml".format(name)
    if not path.exists():
        raise FileNotFoundError("unknown experiment {0}; expected {1}".format(name, path))
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["name"] = config.get("name", name)
    for run in config["runs"]:
        run["checkpoint"] = str((ROOT / run["checkpoint"]).resolve())
    return config


def ensure_checkpoint(path: str) -> str:
    """Return a local .pt path, extracting it from checkpoints/archives when needed."""
    target = Path(path)
    if not target.is_absolute():
        target = (ROOT / target).resolve()
    if target.exists():
        return str(target)
    try:
        member = target.resolve().relative_to(CHECKPOINT_DIR.resolve()).as_posix()
    except ValueError as exc:
        raise FileNotFoundError(target) from exc
    if ".." in member.split("/"):
        raise FileNotFoundError(target)
    for archive in ARCHIVES:
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as bundle:
            if member not in bundle.namelist():
                continue
            bundle.extract(member, path=str(CHECKPOINT_DIR))
            return str(target)
    raise FileNotFoundError(
        "checkpoint {0} is not on disk and not in checkpoints/archives".format(member)
    )


def load_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load a released checkpoint.

    The files are PyTorch pickles. Only load checkpoints you trust. ``weights_only``
    is disabled because the published files store a dictionary around ``model_param``.
    """
    path = ensure_checkpoint(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_param" in checkpoint:
        state = checkpoint["model_param"]
    else:
        state = checkpoint
    model.load_state_dict(state)

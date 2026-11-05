"""Load published weights and hyper-parameters."""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import torch
import yaml

from pep2ccs.paths import CHECKPOINT_DIR, CONFIG_DIR, ROOT

# Released weights are release assets, not files in the git tree.
# Members use paths relative to checkpoints/, such as exp2/lstm.pt.
RELEASE_TAG = "v1.0.0"
RELEASE_URL = "https://github.com/xfcui/PEP2CCS/releases/download/{0}".format(RELEASE_TAG)
ARCHIVE_MEMBERS = {
    "pep2ccs.zip": ("pep2ccs.pt",),
    "exp2.zip": ("exp2/lstm.pt", "exp2/pep2ccs.pt"),
    "exp3.zip": ("exp3/model1.pt", "exp3/model2.pt", "exp3/model3.pt"),
    "exp4.zip": ("exp4/model.pt",),
    "exp5-part1.zip": ("exp5/model1.pt", "exp5/model2.pt", "exp5/model3.pt"),
    "exp5-part2.zip": ("exp5/model4.pt", "exp5/model5.pt", "exp5/model6.pt"),
    "exp5-part3.zip": ("exp5/model7.pt",),
}

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


def _archive_for(member: str) -> Path:
    for name, members in ARCHIVE_MEMBERS.items():
        if member in members:
            return CHECKPOINT_DIR / "archives" / name
    raise FileNotFoundError(
        "checkpoint {0} is not part of release {1}".format(member, RELEASE_TAG)
    )


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pep2ccs"})
    try:
        with urllib.request.urlopen(request) as response, open(temporary, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def ensure_checkpoint(path: str) -> str:
    """Return a local .pt path, downloading its release archive when needed."""
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
    archive = _archive_for(member)
    if not archive.exists():
        url = "{0}/{1}".format(RELEASE_URL, archive.name)
        try:
            _download(url, archive)
        except Exception as exc:
            raise FileNotFoundError(
                "could not download {0} from {1}".format(archive.name, url)
            ) from exc
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise FileNotFoundError(
                "checkpoint {0} is not in {1}".format(member, archive.name)
            )
        bundle.extract(member, path=str(CHECKPOINT_DIR))
    return str(target)


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

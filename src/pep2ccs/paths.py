"""Repository paths. Scripts resolve files from the repo root, not the process cwd."""

from pathlib import Path


def find_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "configs" / "model.yaml").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


ROOT = find_root()
DATA_DIR = ROOT / "data" / "examples"
EXAMPLE_CSV = DATA_DIR / "test_data.csv"
EXAMPLE_ZIP = DATA_DIR / "test_data.zip"
CHECKPOINT_DIR = ROOT / "checkpoints"
CONFIG_DIR = ROOT / "configs"
RESULT_DIR = ROOT / "results"

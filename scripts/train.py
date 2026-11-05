#!/usr/bin/env python
"""Train from a peptide CSV. The loop is reconstructed from configs/model.yaml."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pep2ccs.engine.train import main  # noqa: E402

if __name__ == "__main__":
    main()

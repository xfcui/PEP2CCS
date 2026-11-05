#!/usr/bin/env python
"""Predict CCS. Run from anywhere; paths are resolved from the repository root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pep2ccs.engine.predict import main  # noqa: E402

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reproduce Exp1–Exp5. Example: python scripts/run_experiment.py exp1 --limit 128"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pep2ccs.engine.experiment import main  # noqa: E402

if __name__ == "__main__":
    main()

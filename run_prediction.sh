#!/usr/bin/env bash
# Predict CCS for a peptide CSV.
# Usage: bash run_prediction.sh /path/to/peptides.csv
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

DATA_PATH="${1:-example}"
OUTPUT="${2:-results/predict.csv}"
python scripts/predict.py --data "$DATA_PATH" --output "$OUTPUT"

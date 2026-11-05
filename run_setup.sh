#!/usr/bin/env bash
# Create the PEP2CCS environment and install pinned dependencies.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda, then rerun this script."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate PEP2CCS
pip install -e .
export PYTHONPATH=src
python - << 'PY'
from pep2ccs.data.dataset import ensure_example_csv
print(ensure_example_csv())
PY
echo "Environment PEP2CCS is ready."

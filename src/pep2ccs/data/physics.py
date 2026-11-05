"""Charge-specific linear baseline ``a * charge * sqrt(m/z) + b``.

Coefficients live in ``configs/physics.yaml``. They are the constants that were
hard-coded in the original experiment scripts. The network predicts a residual
in log space and adds ``log(baseline)``.
"""

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml

from pep2ccs.paths import CONFIG_DIR

Coefficient = Tuple[float, float]


def load_fit_params() -> Dict[str, Dict[int, Coefficient]]:
    with open(CONFIG_DIR / "physics.yaml", "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    params = {}
    for name, table in raw.items():
        params[name] = {int(charge): (float(pair[0]), float(pair[1])) for charge, pair in table.items()}
    return params


FIT_PARAMS = load_fit_params()


def baseline_ccs(mz: Sequence[float], charge: Sequence[int], name: str = "default") -> np.ndarray:
    try:
        params: Mapping[int, Coefficient] = FIT_PARAMS[name]
    except KeyError as exc:
        raise ValueError("unknown physics parameter set: {0}".format(name)) from exc

    mz_arr = np.asarray(mz, dtype=np.float64).reshape(-1)
    charge_arr = np.asarray(charge).reshape(-1).astype(int)
    if mz_arr.shape != charge_arr.shape:
        raise ValueError("m/z and charge must have the same number of rows")
    if np.any(mz_arr <= 0):
        raise ValueError("m/z must be positive")

    predicted = np.empty(mz_arr.shape, dtype=np.float64)
    missing = sorted(set(charge_arr.tolist()) - set(params))
    if missing:
        raise ValueError("no physical baseline for charge(s) {0}; supported charges are 2, 3, 4".format(missing))
    for value, (slope, intercept) in params.items():
        mask = charge_arr == value
        predicted[mask] = slope * value * np.sqrt(mz_arr[mask]) + intercept
    if np.any(predicted <= 0):
        raise ValueError("physical baseline is not positive; check m/z and charge")
    return predicted

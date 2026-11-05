"""Metrics reported by the original experiment scripts.

Relative deviation follows the published definition ``(CCS / Pred CCS) * 100 - 100``.
The median of its absolute value is what the Exp1 figure annotated as MAPE.
That number is computed from the predictions; it is not a constant.
"""

from typing import Dict

import numpy as np


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "true and predicted CCS differ in length: {0} vs {1}".format(y_true.size, y_pred.size)
        )
    if y_true.size == 0:
        raise ValueError("no rows to score")
    absolute = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = (y_true / y_pred) * 100.0 - 100.0
    relative = relative[np.isfinite(relative)]
    return {
        "n": int(y_true.size),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(absolute)),
        "max_absolute_error": float(np.max(absolute)),
        "median_abs_relative_deviation": float(np.median(np.abs(relative))) if relative.size else float("nan"),
        "max_relative_deviation": float(np.max(relative)) if relative.size else float("nan"),
    }

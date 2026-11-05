"""Score a peptide table with one published model."""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pep2ccs.data.dataset import PeptideCCSDataset
from pep2ccs.utils.metrics import regression_metrics


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict_dataset(
    model: torch.nn.Module,
    dataset: PeptideCCSDataset,
    device: torch.device,
    batch_size: int,
    progress: bool = True,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    iterator = tqdm(loader, desc="predict", unit="batch") if progress else loader
    columns = []
    model.eval()
    for batch in iterator:
        batch = _move(batch, device)
        with torch.no_grad():
            prediction = model(batch)
        columns.append(torch.exp(prediction).detach().cpu().numpy().reshape(-1))
    if not columns:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(columns).astype(np.float64)


def score_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    vocab_order: str,
    physics: str,
    with_composition: bool,
    length_offset: int,
    progress: bool = True,
) -> Dict:
    dataset = PeptideCCSDataset(
        frame,
        vocab_order=vocab_order,
        physics=physics,
        with_composition=with_composition,
        length_offset=length_offset,
        require_ccs="CCS" in frame.columns,
    )
    predicted = predict_dataset(model, dataset, device, batch_size, progress=progress)
    result = frame.copy()
    result["Pred CCS"] = predicted
    metrics = None
    if "CCS" in frame.columns:
        metrics = regression_metrics(frame["CCS"].to_numpy(), predicted)
        result["Relative deviation"] = (result["CCS"] / result["Pred CCS"]) * 100.0 - 100.0
        result["Absolute error"] = np.abs(result["CCS"] - result["Pred CCS"])
    return {"frame": result, "metrics": metrics}


def save_scatter(frame: pd.DataFrame, path: str, metrics: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charges = sorted(frame["Charge"].unique())
    colors = ["#606060", "#82093B", "#34183E", "#4D779B"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for index, charge in enumerate(charges):
        subset = frame[frame["Charge"] == charge]
        axes[0].scatter(
            subset["m/z"],
            subset["Pred CCS"],
            s=8,
            alpha=0.6,
            c=colors[index % len(colors)],
            label="charge {0}".format(charge),
        )
    axes[0].set_xlabel("m/z")
    axes[0].set_ylabel("Pred CCS")
    axes[0].legend()
    axes[1].scatter(frame["m/z"], frame["CCS"], s=8, alpha=0.35, c="#606060", label="CCS")
    axes[1].scatter(frame["m/z"], frame["Pred CCS"], s=8, alpha=0.35, c="#82093B", label="Pred CCS")
    axes[1].set_xlabel("m/z")
    axes[1].set_ylabel("CCS")
    axes[1].legend()
    axes[2].hist(frame["Relative deviation"], bins=50, color="#34183E", alpha=0.85)
    axes[2].set_xlim(-10, 10)
    axes[2].set_xlabel("relative deviation (%)")
    axes[2].set_ylabel("count")
    if metrics is not None:
        axes[2].text(
            0.98,
            0.95,
            "median |rel| {0:.3f}%".format(metrics["median_abs_relative_deviation"]),
            transform=axes[2].transAxes,
            ha="right",
            va="top",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

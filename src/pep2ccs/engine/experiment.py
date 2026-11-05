"""Run one published experiment configuration."""

import argparse
import os

import pandas as pd

from pep2ccs.data.dataset import ensure_example_csv, read_peptide_table
from pep2ccs.data.subsets import select_subset
from pep2ccs.engine.checkpoint import load_experiment, load_hparams, load_weights
from pep2ccs.engine.evaluate import save_scatter, score_frame
from pep2ccs.model.factory import build_model
from pep2ccs.paths import RESULT_DIR
from pep2ccs.utils.device import get_device


def run_experiment(name: str, data_path: str = None, batch_size: int = None, device: str = None, limit: int = None, plot: bool = None):
    config = load_experiment(name)
    hparams = load_hparams()
    source = data_path or ensure_example_csv()
    frame = read_peptide_table(source, limit=limit)
    device_obj = get_device(device)
    output_dir = RESULT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in config["runs"]:
        subset_name = run.get("subset", "full")
        subset = select_subset(frame, subset_name)
        if len(subset) == 0:
            print("{0} [{1}] skipped: no rows".format(run["name"], subset_name))
            rows.append({"run": run["name"], "subset": subset_name, "checkpoint": run["checkpoint"], "n": 0})
            continue
        model = build_model(run["model"], hparams, dropout=0.0, regress_dropout=0.0).to(device_obj)
        load_weights(model, run["checkpoint"], device_obj)
        scored = score_frame(
            model,
            subset,
            device_obj,
            batch_size or int(hparams["batch_size"]),
            vocab_order=run.get("vocab", config.get("vocab", "standard")),
            physics=run.get("physics", config.get("physics", "default")),
            with_composition=bool(run.get("composition", config.get("composition", False))),
            length_offset=int(run.get("length_offset", config.get("length_offset", 0))),
        )
        prediction_path = output_dir / "{0}_{1}.csv".format(run["name"], subset_name)
        scored["frame"].to_csv(prediction_path, index=False)
        metrics = scored["metrics"] or {}
        record = {"run": run["name"], "subset": subset_name, "checkpoint": run["checkpoint"], "predictions": str(prediction_path)}
        record.update(metrics)
        rows.append(record)
        if metrics:
            print(
                "{run} [{subset}] n={n} MSE={mse:.6f} median |rel|={median_abs_relative_deviation:.4f}%".format(
                    run=run["name"], subset=subset_name, **metrics
                )
            )
        draw = config.get("plot", False) if plot is None else plot
        if draw and metrics and "m/z" in scored["frame"].columns:
            save_scatter(scored["frame"], str(output_dir / "{0}_{1}.png".format(run["name"], subset_name)), metrics)
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "metrics.csv"
    summary.to_csv(summary_path, index=False)
    print("metrics: {0}".format(summary_path))
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce a PEP2CCS experiment")
    parser.add_argument("experiment", choices=["exp1", "exp2", "exp3", "exp4", "exp5"])
    parser.add_argument("--data", default=None, help="defaults to the bundled example CSV")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=None)
    args = parser.parse_args(argv)
    run_experiment(args.experiment, args.data, args.batch_size, args.device, args.limit, args.plot)


if __name__ == "__main__":
    main()

"""Predict CCS for a CSV and, when CCS is present, print the error."""

import argparse
import os

import pandas as pd

from pep2ccs.data.dataset import ensure_example_csv, read_peptide_table
from pep2ccs.engine.checkpoint import load_hparams, load_weights
from pep2ccs.engine.evaluate import save_scatter, score_frame
from pep2ccs.model.factory import build_model
from pep2ccs.paths import RESULT_DIR, ROOT
from pep2ccs.utils.device import get_device


def predict_file(
    data_path: str,
    checkpoint: str,
    output: str,
    model_name: str = "pep2ccs",
    batch_size: int = None,
    device: str = None,
    limit: int = None,
    plot: bool = False,
) -> pd.DataFrame:
    hparams = load_hparams()
    frame = read_peptide_table(data_path, limit=limit)
    device_obj = get_device(device)
    model = build_model(model_name, hparams, dropout=0.0, regress_dropout=0.0).to(device_obj)
    load_weights(model, checkpoint, device_obj)
    scored = score_frame(
        model,
        frame,
        device_obj,
        batch_size or int(hparams["batch_size"]),
        vocab_order="standard",
        physics="default",
        with_composition=False,
        length_offset=0,
    )
    directory = os.path.dirname(os.path.abspath(output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    scored["frame"].to_csv(output, index=False)
    if scored["metrics"] is not None:
        metrics = scored["metrics"]
        print(
            "n={n}  MSE={mse:.6f}  MAE={mae:.6f}  median |relative deviation|={median_abs_relative_deviation:.4f}%".format(
                **metrics
            )
        )
    else:
        print("CCS column absent; wrote predictions only")
    print("predictions: {0}".format(output))
    if plot and "CCS" in frame.columns:
        figure = os.path.splitext(output)[0] + ".png"
        save_scatter(scored["frame"], figure, scored["metrics"])
        print("figure: {0}".format(figure))
    return scored["frame"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict peptide CCS with PEP2CCS")
    parser.add_argument("--data", required=True, help="CSV with Sequence, Charge, and m/z")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "pep2ccs.pt"),
        help="checkpoint containing model_param",
    )
    parser.add_argument("--output", default=str(RESULT_DIR / "predict.csv"))
    parser.add_argument("--model", default="pep2ccs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    data = args.data
    if data == "example":
        data = ensure_example_csv()
    predict_file(
        data,
        args.checkpoint,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()

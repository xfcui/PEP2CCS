"""Train PEP2CCS from a peptide CSV.

The original repository published weights and hyper-parameters but not the
training loop. This entry point follows those hyper-parameters:

- Adam with the learning rate in ``configs/model.yaml``
- ``StepLR`` drops the learning rate by 10 at epoch ``reduce_lr_step``
- the loss is MSE in log CCS space, which is the space the network emits
- validation is the ``valid_fraction`` split with ``split_seed``
- the best validation checkpoint is kept, and training stops after
  ``early_stop`` epochs without improvement

Treat a newly trained checkpoint as a rerun, not as a bit-identical copy of
``checkpoints/pep2ccs.pt``, until the original optimization code is recovered.
"""

import argparse
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from pep2ccs.data.dataset import PeptideCCSDataset, read_peptide_table
from pep2ccs.engine.checkpoint import load_hparams
from pep2ccs.model.factory import build_model
from pep2ccs.utils.device import get_device
from pep2ccs.utils.seed import set_seed


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _epoch_loss(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        batch = _move(batch, device)
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(prediction, batch["ccs"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * batch["ccs"].shape[0]
        count += int(batch["ccs"].shape[0])
    return total / max(count, 1)


def train_model(
    train_csv: str,
    output: str,
    model_name: str = "pep2ccs",
    device: str = None,
    epochs: int = None,
    batch_size: int = None,
):
    hparams = load_hparams()
    set_seed(int(hparams["split_seed"]))
    frame = read_peptide_table(train_csv)
    if "CCS" not in frame.columns:
        raise ValueError("training requires a CCS column")
    train_frame, valid_frame = train_test_split(
        frame,
        test_size=float(hparams["valid_fraction"]),
        random_state=int(hparams["split_seed"]),
    )
    train_frame = train_frame.reset_index(drop=True)
    valid_frame = valid_frame.reset_index(drop=True)
    dataset_kwargs = {}
    if model_name == "exp4":
        dataset_kwargs = {"vocab_order": "alphabetical", "physics": "exp4", "with_composition": True, "length_offset": 3}
    elif model_name == "lstm":
        dataset_kwargs = {}
    train_data = PeptideCCSDataset(train_frame, require_ccs=True, **dataset_kwargs)
    valid_data = PeptideCCSDataset(valid_frame, require_ccs=True, **dataset_kwargs)
    batch_size = batch_size or int(hparams["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    device_obj = get_device(device)
    model = build_model(model_name, hparams).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(hparams["reduce_lr_step"]),
        gamma=0.1,
    )
    best = float("inf")
    stale = 0
    max_epochs = int(hparams["epochs"] if epochs is None else epochs)
    patience = int(hparams["early_stop"])
    directory = os.path.dirname(os.path.abspath(output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    for epoch in tqdm(range(1, max_epochs + 1), desc="epoch"):
        train_loss = _epoch_loss(model, train_loader, device_obj, optimizer)
        valid_loss = _epoch_loss(model, valid_loader, device_obj, optimizer=None)
        scheduler.step()
        if epoch % int(hparams["eval_interval"]) == 0 or epoch == 1:
            print("epoch {0}  train_log_mse {1:.6f}  valid_log_mse {2:.6f}".format(epoch, train_loss, valid_loss))
        if valid_loss < best:
            best = valid_loss
            stale = 0
            torch.save(
                {
                    "model_param": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best,
                    "model_name": model_name,
                    "hparams": hparams,
                },
                output,
            )
        else:
            stale += 1
            if stale >= patience:
                print("early stop at epoch {0}".format(epoch))
                break
    print("best valid log-MSE {0:.6f}; checkpoint {1}".format(best, output))
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a PEP2CCS model")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="pep2ccs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)
    train_model(args.train_csv, args.output, args.model, args.device, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()

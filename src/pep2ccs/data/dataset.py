"""Read a peptide table once and feed every experiment from it."""

import zipfile
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pep2ccs.data.encoding import MAX_LEN, build_vocab, composition_vector, encode_sequence
from pep2ccs.data.physics import baseline_ccs
from pep2ccs.paths import EXAMPLE_CSV, EXAMPLE_ZIP

REQUIRED_COLUMNS = ("Sequence", "Charge", "m/z")


def ensure_example_csv() -> str:
    """Extract the bundled example table next to its zip archive."""
    if EXAMPLE_CSV.exists():
        return str(EXAMPLE_CSV)
    if not EXAMPLE_ZIP.exists():
        raise FileNotFoundError("missing example data: {0}".format(EXAMPLE_ZIP))
    EXAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EXAMPLE_ZIP) as archive:
        archive.extract("test_data.csv", path=str(EXAMPLE_CSV.parent))
    if not EXAMPLE_CSV.exists():
        raise FileNotFoundError("test_data.zip did not contain test_data.csv")
    return str(EXAMPLE_CSV)


def read_peptide_table(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "{0} is missing column(s) {1}. Required columns are Sequence, Charge, and m/z. "
            "CCS is required for evaluation, and Length is optional.".format(path, ", ".join(missing))
        )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        frame = frame.iloc[:limit].copy()
    frame = frame.reset_index(drop=True)
    if "Length" not in frame.columns:
        frame["Length"] = frame["Sequence"].astype(str).str.len()
    return frame


class PeptideCCSDataset(Dataset):
    """One row is the tensors consumed by every published PEP2CCS variant.

    ``length_offset`` is 3 for Exp4, which stored peptide length plus the three
    special tokens. Other experiments use the length column as published.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        vocab_order: str = "standard",
        physics: str = "default",
        with_composition: bool = False,
        length_offset: int = 0,
        require_ccs: bool = False,
        max_len: int = MAX_LEN,
    ):
        self.vocab = build_vocab(vocab_order)
        self.max_len = max_len
        self.with_composition = with_composition
        sequences = frame["Sequence"].astype(str).tolist()
        charges = frame["Charge"].astype(int).to_numpy()
        if "Length" in frame.columns:
            lengths = frame["Length"].astype(int).to_numpy()
        else:
            lengths = np.asarray([len(sequence) for sequence in sequences], dtype=int)
        lengths = lengths + int(length_offset)
        mz = frame["m/z"].astype(float).to_numpy()
        encoded = [
            encode_sequence(sequence, charge, self.vocab, max_len=max_len)
            for sequence, charge in zip(sequences, charges)
        ]
        self.seq = torch.tensor(np.asarray(encoded, dtype=np.int64), dtype=torch.long)
        self.charge = torch.tensor(charges.reshape(-1, 1), dtype=torch.long)
        self.length = torch.tensor(lengths, dtype=torch.long)
        self.mz = torch.tensor(mz.reshape(-1, 1), dtype=torch.float32)
        ccs2 = np.log(baseline_ccs(mz, charges, physics)).astype(np.float32).reshape(-1, 1)
        self.ccs2 = torch.tensor(ccs2, dtype=torch.float32)
        if "CCS" in frame.columns:
            observed = frame["CCS"].astype(float).to_numpy()
            if np.any(observed <= 0):
                raise ValueError("CCS must be positive")
            self.ccs = torch.tensor(np.log(observed).astype(np.float32).reshape(-1, 1), dtype=torch.float32)
            self.has_ccs = True
        else:
            if require_ccs:
                raise ValueError("evaluation requires a CCS column")
            self.ccs = torch.zeros((len(frame), 1), dtype=torch.float32)
            self.has_ccs = False
        if with_composition:
            vectors = [composition_vector(sequence, charge) for sequence, charge in zip(sequences, charges)]
            self.vector = torch.tensor(np.asarray(vectors, dtype=np.float32), dtype=torch.float32)
        else:
            self.vector = None

    def __len__(self) -> int:
        return int(self.seq.shape[0])

    def __getitem__(self, index: int):
        item = {
            "seq": self.seq[index],
            "ccs": self.ccs[index],
            "charge": self.charge[index],
            "length": self.length[index],
            "mz": self.mz[index],
            "ccs2": self.ccs2[index],
        }
        if self.vector is not None:
            item["vector"] = self.vector[index]
        return item

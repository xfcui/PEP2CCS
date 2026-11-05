"""Tokenization used by the published PEP2CCS checkpoints.

Two vocabularies exist because the experiments were not trained with the same
amino-acid order:

- ``standard`` is the order used by the main model and Exp1–Exp3 and Exp5.
- ``alphabetical`` is the order produced by sorting the 20 standard residues,
  which matches Exp4 when every standard residue occurs in the table.
"""

from typing import Dict, List, Sequence

MAX_LEN = 64
VOCAB_SIZE = 32
STANDARD_AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")
ALPHABETICAL_AMINO_ACIDS = sorted(STANDARD_AMINO_ACIDS)
SUPPORTED_CHARGES = (2, 3, 4)

_ORDERS = {
    "standard": STANDARD_AMINO_ACIDS,
    "alphabetical": ALPHABETICAL_AMINO_ACIDS,
}


def build_vocab(order: str = "standard") -> Dict[str, int]:
    try:
        amino_acids = _ORDERS[order]
    except KeyError as exc:
        raise ValueError("vocab order must be 'standard' or 'alphabetical'") from exc
    vocab = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "Charge2": 3,
        "Charge3": 4,
        "Charge4": 5,
    }
    for index, residue in enumerate(amino_acids):
        vocab[residue] = index + 6
    return vocab


def encode_sequence(sequence: str, charge: int, vocab: Dict[str, int], max_len: int = MAX_LEN) -> List[int]:
    """Encode one peptide as ``[SOS, charge, residues..., EOS, PAD...]``."""
    charge = int(charge)
    if charge not in SUPPORTED_CHARGES:
        raise ValueError(
            "charge must be 2, 3, or 4; got {0} for sequence {1}".format(charge, sequence)
        )
    unknown = sorted({char for char in sequence if char not in vocab})
    if unknown:
        raise ValueError(
            "sequence {0} contains residues outside the vocabulary: {1}".format(sequence, "".join(unknown))
        )
    tokens = [vocab["<SOS>"], vocab["Charge{0}".format(charge)]]
    tokens.extend(vocab[char] for char in sequence)
    tokens.append(vocab["<EOS>"])
    if len(tokens) > max_len:
        raise ValueError(
            "sequence {0} needs {1} tokens including specials, above max_len {2}".format(
                sequence, len(tokens), max_len
            )
        )
    tokens.extend([vocab["<PAD>"]] * (max_len - len(tokens)))
    return tokens


def encode_row(sequence: str, charge: int, order: str = "standard", max_len: int = MAX_LEN) -> List[int]:
    return encode_sequence(sequence, charge, build_vocab(order), max_len=max_len)


def composition_vector(sequence: str, charge: int) -> List[float]:
    """Residue counts in the main-model amino-acid order, followed by a charge one-hot.

    This is the 23-dimensional vector consumed by the Exp4 checkpoint.
    """
    charge = int(charge)
    if charge not in SUPPORTED_CHARGES:
        raise ValueError("charge must be 2, 3, or 4; got {0}".format(charge))
    counts = [0] * len(STANDARD_AMINO_ACIDS)
    index = {residue: pos for pos, residue in enumerate(STANDARD_AMINO_ACIDS)}
    unknown = sorted({char for char in sequence if char not in index})
    if unknown:
        raise ValueError(
            "sequence {0} contains residues outside the vocabulary: {1}".format(sequence, "".join(unknown))
        )
    for char in sequence:
        counts[index[char]] += 1
    one_hot = [0, 0, 0]
    one_hot[charge - 2] = 1
    return [float(value) for value in counts + one_hot]


def amino_acid_order(order: str) -> Sequence[str]:
    return tuple(_ORDERS[order])

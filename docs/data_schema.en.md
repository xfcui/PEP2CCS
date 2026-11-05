<p align="right"><a href="../README.md">README</a> · <a href="data_schema.md">中文</a> | <b>English</b></p>

# Data schema

`data/examples/test_data.zip` expands to about 210,000 rows. Prediction uses the columns below. Every other column is copied through to the output.

| Column | Required | Meaning |
|---|---|---|
| Sequence | yes | Unmodified peptide sequence, 20 standard amino acids only |
| Charge | yes | 2, 3, or 4 |
| m/z | yes | Mass-to-charge ratio, must be positive |
| Length | no | Number of residues. Defaults to `len(Sequence)` |
| CCS | evaluation and training | Observed collision cross section, must be positive |

`Modified_sequence` is kept as a record. Encoding uses `Sequence` after modification labels have been removed.

## Vocabulary

The main model and Exp1, Exp2, Exp3, and Exp5 use the `standard` order `ARNDCQEGHILKMFPSTWYV`.

| token | id |
|---|---|
| PAD | 0 |
| SOS | 1 |
| EOS | 2 |
| Charge2 / Charge3 / Charge4 | 3 / 4 / 5 |
| amino acids | from 6, in the order above |

One sample is `[SOS, Charge, residues..., EOS, PAD...]` with length 64, so a peptide can have at most 61 residues. Unknown characters, other charges, and sequences that do not fit raise an error.

Exp4 orders the 20 amino acids alphabetically, because the original script built the vocabulary with `sorted`. The example table contains all 20 residues, so this fixed vocabulary assigns the same ids as sorting the test table. Exp4 also adds 3 to the length and appends a 20-dimensional residue count plus a 3-dimensional charge one-hot.

## Physical baseline

`configs/physics.yaml`:

`baseline = a * charge * sqrt(m/z) + b`

The network output is added to `log(baseline)`, and the exponential of that sum is the CCS. Exp4 uses the `exp4` coefficients in the file. The other experiments use `default`. How each experiment uses the coefficients is in the [experiment notes](experiments.en.md).

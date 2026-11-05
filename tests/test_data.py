"""Encoding, physics, and subset tests. These do not need a GPU or a checkpoint."""

import unittest

import numpy as np
import pandas as pd

from pep2ccs.data.encoding import (
    ALPHABETICAL_AMINO_ACIDS,
    STANDARD_AMINO_ACIDS,
    build_vocab,
    composition_vector,
    encode_sequence,
)
from pep2ccs.data.physics import FIT_PARAMS, baseline_ccs
from pep2ccs.data.subsets import select_subset


class EncodingTest(unittest.TestCase):
    def test_standard_vocab_matches_the_published_order(self):
        vocab = build_vocab("standard")
        self.assertEqual(vocab["<PAD>"], 0)
        self.assertEqual(vocab["<SOS>"], 1)
        self.assertEqual(vocab["<EOS>"], 2)
        self.assertEqual(vocab["Charge2"], 3)
        self.assertEqual([vocab[residue] for residue in STANDARD_AMINO_ACIDS], list(range(6, 26)))

    def test_alphabetical_vocab_differs_at_cysteine(self):
        standard = build_vocab("standard")
        alphabetical = build_vocab("alphabetical")
        self.assertEqual(alphabetical["A"], 6)
        self.assertEqual(alphabetical["C"], 7)
        self.assertNotEqual(standard["C"], alphabetical["C"])
        self.assertEqual(ALPHABETICAL_AMINO_ACIDS[0], "A")

    def test_encode_specials_and_padding(self):
        vocab = build_vocab("standard")
        tokens = encode_sequence("AC", 2, vocab)
        self.assertEqual(len(tokens), 64)
        self.assertEqual(tokens[:5], [1, 3, vocab["A"], vocab["C"], 2])
        self.assertTrue(all(token == 0 for token in tokens[5:]))

    def test_rejects_unknown_residue_charge_and_overlong_peptide(self):
        vocab = build_vocab("standard")
        with self.assertRaises(ValueError):
            encode_sequence("AX", 2, vocab)
        with self.assertRaises(ValueError):
            encode_sequence("A", 5, vocab)
        with self.assertRaises(ValueError):
            encode_sequence("A" * 62, 2, vocab)

    def test_composition_vector_is_counts_plus_charge_one_hot(self):
        vector = composition_vector("AAC", 4)
        self.assertEqual(len(vector), 23)
        self.assertEqual(vector[STANDARD_AMINO_ACIDS.index("A")], 2.0)
        self.assertEqual(vector[STANDARD_AMINO_ACIDS.index("C")], 1.0)
        self.assertEqual(vector[-3:], [0.0, 0.0, 1.0])


class PhysicsTest(unittest.TestCase):
    def test_published_coefficients_are_loaded(self):
        self.assertEqual(FIT_PARAMS["default"][2][0], 7.501715950104645)
        self.assertEqual(FIT_PARAMS["exp4"][4][1], 175.14669422777405)

    def test_baseline_uses_charge_specific_line(self):
        slope, intercept = FIT_PARAMS["default"][2]
        value = baseline_ccs([100.0], [2], "default")
        self.assertAlmostEqual(float(value[0]), slope * 2 * np.sqrt(100.0) + intercept)

    def test_unknown_charge_is_rejected(self):
        with self.assertRaises(ValueError):
            baseline_ccs([100.0], [5], "default")


class SubsetTest(unittest.TestCase):
    def test_multi_charge_keeps_sequences_seen_with_more_than_one_charge(self):
        frame = pd.DataFrame(
            {
                "Sequence": ["AA", "AA", "BB", "CC", "CC"],
                "Charge": [2, 3, 2, 4, 4],
            }
        )
        subset = select_subset(frame, "multi_charge")
        self.assertEqual(subset["Sequence"].tolist(), ["AA", "AA"])


if __name__ == "__main__":
    unittest.main()

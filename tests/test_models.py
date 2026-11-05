"""Forward shapes and checkpoint compatibility."""

import math
import unittest
from pathlib import Path

import pandas as pd
import torch

from pep2ccs.data.dataset import PeptideCCSDataset
from pep2ccs.engine.checkpoint import ensure_checkpoint, load_hparams, load_weights
from pep2ccs.model.factory import build_model, model_names
from pep2ccs.paths import ROOT
from pep2ccs.utils.metrics import regression_metrics


def _batch(with_vector=False, batch_size=2):
    seq = torch.randint(1, 26, (batch_size, 64))
    seq[:, 20:] = 0
    seq[:, 0] = 1
    batch = {
        "seq": seq,
        "ccs": torch.zeros(batch_size, 1),
        "charge": torch.tensor([[2], [3]][:batch_size], dtype=torch.long),
        "length": torch.tensor([8, 9][:batch_size], dtype=torch.long),
        "mz": torch.tensor([[400.0], [500.0]][:batch_size]),
        "ccs2": torch.zeros(batch_size, 1),
    }
    if with_vector:
        batch["vector"] = torch.zeros(batch_size, 23)
    return batch


class ModelShapeTest(unittest.TestCase):
    def test_every_variant_returns_one_ccs_per_row(self):
        hparams = {
            "embedding_size": 16,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "max_len": 64,
        }
        for name in model_names():
            model = build_model(name, hparams, dropout=0.0, regress_dropout=0.0)
            model.eval()
            batch = _batch(with_vector=(name == "exp4"))
            with torch.no_grad():
                output = model(batch)
            self.assertEqual(tuple(output.shape), (2, 1), name)

    def test_eval_predictions_do_not_depend_on_batch_size(self):
        hparams = {
            "embedding_size": 16,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "max_len": 64,
        }
        model = build_model("pep2ccs", hparams, dropout=0.0, regress_dropout=0.0)
        model.eval()
        batch = _batch()
        with torch.no_grad():
            together = model(batch)
            apart = torch.cat([model({key: value[:1] for key, value in batch.items()}) for _ in range(1)] + [
                model({key: value[1:] for key, value in batch.items()})
            ])
        self.assertTrue(torch.allclose(together, apart, atol=1e-5))


class DatasetTest(unittest.TestCase):
    def test_missing_length_is_taken_from_the_sequence(self):
        frame = pd.DataFrame({"Sequence": ["AC"], "Charge": [2], "m/z": [100.0], "CCS": [400.0]})
        dataset = PeptideCCSDataset(frame, require_ccs=True)
        self.assertEqual(int(dataset.length[0]), 2)
        self.assertAlmostEqual(float(dataset.ccs[0]), math.log(400.0), places=5)
        self.assertTrue(math.isfinite(float(dataset.ccs2[0])))


class MetricTest(unittest.TestCase):
    def test_relative_deviation_uses_the_published_ratio(self):
        metrics = regression_metrics([110.0], [100.0])
        self.assertAlmostEqual(metrics["median_abs_relative_deviation"], 10.0)
        self.assertAlmostEqual(metrics["mse"], 100.0)


class CheckpointTest(unittest.TestCase):
    def test_published_checkpoints_match_the_architectures(self):
        hparams = load_hparams()
        expected = {
            "pep2ccs": ROOT / "checkpoints" / "pep2ccs.pt",
            "lstm": ROOT / "checkpoints" / "exp2" / "lstm.pt",
            "exp3_mz_embed": ROOT / "checkpoints" / "exp3" / "model1.pt",
            "exp3_length_embed": ROOT / "checkpoints" / "exp3" / "model2.pt",
            "exp3_joint_embed": ROOT / "checkpoints" / "exp3" / "model3.pt",
            "exp4": ROOT / "checkpoints" / "exp4" / "model.pt",
            "exp5_cls": ROOT / "checkpoints" / "exp5" / "model1.pt",
            "exp5_cls_max": ROOT / "checkpoints" / "exp5" / "model2.pt",
            "exp5_max": ROOT / "checkpoints" / "exp5" / "model3.pt",
            "exp5_cnn_max": ROOT / "checkpoints" / "exp5" / "model4.pt",
            "exp5_mean": ROOT / "checkpoints" / "exp5" / "model5.pt",
            "exp5_mean_max": ROOT / "checkpoints" / "exp5" / "model6.pt",
            "exp5_mean_cnn": ROOT / "checkpoints" / "exp5" / "model7.pt",
        }
        for name, path in expected.items():
            extracted = Path(ensure_checkpoint(str(path)))
            self.assertTrue(extracted.exists(), str(extracted))
            model = build_model(name, hparams, dropout=0.0, regress_dropout=0.0)
            load_weights(model, str(extracted), torch.device("cpu"))
            if name == "pep2ccs":
                model.eval()
                with torch.no_grad():
                    output = model(_batch())
                self.assertEqual(tuple(output.shape), (2, 1))
        exp2 = ROOT / "checkpoints" / "exp2" / "pep2ccs.pt"
        self.assertTrue(exp2.exists(), str(exp2))
        exp2_model = build_model("exp2_pep2ccs", hparams, dropout=0.0, regress_dropout=0.0)
        load_weights(exp2_model, str(exp2), torch.device("cpu"))
        exp2_model.eval()
        with torch.no_grad():
            output = exp2_model(_batch())
        self.assertEqual(tuple(output.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()

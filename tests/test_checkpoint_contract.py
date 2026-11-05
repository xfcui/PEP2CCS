"""Checkpoint tensor shapes that the rebuilt modules must accept.

This test does not import torch. It reads the zip archive PyTorch uses for
``.pt`` files and checks the parameter names and shapes the architectures were
rebuilt around.
"""

import io
import pickle
import unittest
import zipfile
from pathlib import Path

from pep2ccs.engine.checkpoint import ensure_checkpoint
from pep2ccs.paths import ROOT


class _Stub:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _factory(*args, **kwargs):
    return _Stub(*args)


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return ("storage",)

    def find_class(self, module, name):
        if module == "collections" and name in {"OrderedDict", "defaultdict"}:
            return getattr(__import__("collections"), name)
        if module.startswith("torch") or module.startswith("numpy"):
            return _factory
        return super().find_class(module, name)


def _state_dict(path):
    with zipfile.ZipFile(path) as archive:
        payload = [name for name in archive.namelist() if name.endswith("data.pkl")][0]
        checkpoint = _Unpickler(io.BytesIO(archive.read(payload))).load()
    return checkpoint["model_param"]


def _shape(value):
    size = value.args[2]
    if not isinstance(size, tuple):
        raise AssertionError("tensor size missing from checkpoint pickle")
    return size


class CheckpointContractTest(unittest.TestCase):
    def _require(self, relative, expected):
        path = Path(ensure_checkpoint(str(ROOT / relative)))
        self.assertTrue(path.exists(), str(path))
        state = _state_dict(path)
        for key, shape in expected.items():
            self.assertIn(key, state, relative)
            self.assertEqual(_shape(state[key]), shape, "{0}: {1}".format(relative, key))

    def test_main_and_pooling_ablations(self):
        common = {
            "Embedding.seq_embedding.weight": (32, 128),
            "Pooling.local_agg.0.weight": (128, 128, 3),
            "Regression.mlp.0.weight": (128, 258),
            "Regression.mlp.7.weight": (1, 64),
        }
        self._require("checkpoints/pep2ccs.pt", common)
        self._require(
            "checkpoints/exp5/model1.pt",
            {
                "Pooling.local_agg.0.weight": (128, 128, 3),
                "Regression.mlp.0.weight": (128, 130),
            },
        )
        self._require("checkpoints/exp5/model2.pt", {"Regression.mlp.0.weight": (128, 258)})
        self._require("checkpoints/exp5/model7.pt", {
            "Pooling.local_agg.0.weight": (128, 128, 3),
            "Regression.mlp.0.weight": (128, 258),
        })

    def test_exp2_checkpoint_is_not_the_main_architecture(self):
        self._require(
            "checkpoints/exp2/pep2ccs.pt",
            {
                "Embedding.seq_embedding.weight": (50, 128),
                "Pooling.local_agg.0.weight": (128, 128, 3),
                "Regression.mlp.0.weight": (128, 259),
                "Regression.mlp.6.weight": (1, 64),
            },
        )
        state = _state_dict(ROOT / "checkpoints" / "exp2" / "pep2ccs.pt")
        self.assertNotIn("Regression.mlp.7.weight", state)

    def test_exp3_exp4_and_lstm(self):
        self._require(
            "checkpoints/exp3/model1.pt",
            {
                "Embedding.mlp.0.weight": (128, 2),
                "Regression.mlp.0.weight": (128, 257),
            },
        )
        self._require(
            "checkpoints/exp3/model3.pt",
            {
                "Embedding.mlp.0.weight": (128, 3),
                "Regression.mlp.0.weight": (128, 256),
            },
        )
        self._require(
            "checkpoints/exp4/model.pt",
            {
                "Pooling.local_agg2.0.weight": (128, 128, 5),
                "Pooling.global_linear.weight": (1, 128),
                "Regression.mlp.0.weight": (128, 281),
                "Regression.mlp.3.weight": (1, 128),
            },
        )
        self._require(
            "checkpoints/exp2/lstm.pt",
            {
                "Embedding.seq_embedding.weight": (32, 128),
                "Encoder.LSTM.weight_ih_l0": (512, 128),
                "Encoder.LSTM.weight_ih_l1": (512, 256),
                "Regression.mlp.0.weight": (128, 257),
                "Regression.mlp.3.weight": (1, 128),
            },
        )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.train import load_pretrained


class _RecordingModel:
    def __init__(self):
        self.loaded = None

    def load_compatible_state_dict(self, state_dict):
        self.loaded = state_dict
        return SimpleNamespace(missing_keys=[]), []


class LoadPretrainedTest(unittest.TestCase):
    def test_keeps_target_dataset_normalization_buffers(self):
        norm_names = (
            "trans_mean",
            "trans_std",
            "wrist_mean",
            "wrist_std",
            "finger_mean",
            "finger_std",
            "shape_mean",
            "shape_std",
        )
        state = {
            f"module.flow.denoise_fn.{name}": torch.tensor([99.0])
            for name in norm_names
        }
        state["module.flow.denoise_fn.out_translation.weight"] = torch.tensor([1.0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "pretrained.pt"
            torch.save({"model": state}, checkpoint)
            model = _RecordingModel()
            loaded = load_pretrained(model, checkpoint, torch.device("cpu"))

        self.assertEqual(loaded, 1)
        self.assertEqual(
            set(model.loaded), {"flow.denoise_fn.out_translation.weight"}
        )


if __name__ == "__main__":
    unittest.main()

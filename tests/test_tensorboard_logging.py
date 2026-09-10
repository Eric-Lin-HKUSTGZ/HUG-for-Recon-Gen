"""Regression tests for the rank-0 TensorBoard training logger."""

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.train import (
    SummaryWriter,
    _close_tensorboard_writer,
    _create_tensorboard_writer,
    _write_tensorboard_scalars,
)


@unittest.skipIf(SummaryWriter is None, "tensorboard is not installed")
class TensorBoardLoggingTest(unittest.TestCase):
    def test_loss_and_validation_tags_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            train_cfg = OmegaConf.create(
                {
                    "tensorboard": {
                        "enabled": True,
                        "log_dir": "events",
                        "flush_secs": 1,
                        "max_queue": 1,
                    }
                }
            )
            writer, log_dir = _create_tensorboard_writer(
                train_cfg, out_dir, start_step=7
            )
            self.assertEqual(log_dir, out_dir / "events")

            _write_tensorboard_scalars(
                writer,
                "train",
                {
                    "loss": 2.0,
                    "loss_flow_weighted": 1.25,
                    "loss_3d_weighted": 0.75,
                    "loss_mesh_3d_weighted": 0.5,
                    "lv": 1.25,
                    "l3d": 0.011,
                    "l3d_time_weighted": 0.0075,
                    "lmesh": 0.010,
                    "lmesh_time_weighted": 0.006,
                    "train_mpjpe_mm": 8.5,
                    "train_mpvpe_mm": 8.2,
                    "grad_norm": 0.75,
                    "lr": 1e-4,
                },
                step=8,
            )
            _write_tensorboard_scalars(
                writer,
                "validation",
                {
                    "datasets": {
                        "dexycb": {"pa_mpjpe": 5.5, "pa_mpvpe": 5.3}
                    },
                    "score": 5.4,
                },
                step=10,
            )
            _close_tensorboard_writer(writer)

            events = EventAccumulator(str(log_dir))
            events.Reload()
            scalar_tags = set(events.Tags()["scalars"])
            self.assertTrue(
                {
                    "loss/total",
                    "loss_weighted/flow",
                    "loss_weighted/3d_landmarks",
                    "loss_weighted/3d_mesh",
                    "loss_raw/flow",
                    "loss_raw/3d_landmarks",
                    "loss_raw/3d_landmarks_time_weighted",
                    "loss_raw/3d_mesh",
                    "loss_raw/3d_mesh_time_weighted",
                    "error_mm/train_mpjpe",
                    "error_mm/train_mpvpe",
                    "optimization/gradient_norm",
                    "optimization/learning_rate",
                    "validation/dexycb/pa_mpjpe_mm",
                    "validation/dexycb/pa_mpvpe_mm",
                    "validation/selection_score_mm",
                }.issubset(scalar_tags)
            )
            self.assertEqual(events.Scalars("loss/total")[-1].step, 8)
            self.assertAlmostEqual(
                events.Scalars("validation/selection_score_mm")[-1].value,
                5.4,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()

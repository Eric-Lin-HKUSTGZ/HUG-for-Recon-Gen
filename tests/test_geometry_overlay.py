"""Small regression fixtures: no MANO assets, images or remote data required."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataloader.geometry_overlay import ARRAYS, CONVENTION, VERSION, GeometryOverlay


class OverlayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "pkls"
        self.dataset.mkdir()
        self.directory = self.root / "overlay"
        self.directory.mkdir()
        self.manifest = {"version": VERSION, "parameter_convention": CONVENTION,
                         "status": "complete", "dataset_root": str(self.dataset), "count": 2}
        self.write_manifest()
        np.save(self.directory / "samples.npy", np.array(["left_sample", "right_sample"]))
        for key, (shape, dtype) in ARRAYS.items():
            value = np.zeros((2, *shape), dtype=dtype)
            if key == "params":
                value[:, :3] = [0.02, 0.03, 0.6]
                value[:, 3:99] = np.tile(np.eye(3)[:, :2].reshape(6), 16)
                value[:, 99:] = np.arange(10)
            elif key in ("joints", "vertices"):
                value[:] = [0.02, 0.03, 0.6]
            elif key == "camera_K":
                value[:] = [[600, 0, 319], [0, 600, 239], [0, 0, 1]]
            elif key == "source_is_left":
                value[:] = [True, False]
            np.save(self.directory / f"{key}.npy", value)
        for side in ("left", "right"):
            np.save(self.directory / f"faces_{side}.npy", np.array([[0, 1, 2]], dtype=np.int32))
        self.record = {"image": b"RGB preserved", "depth": b"depth preserved",
                       "object_mask": b"old inaccurate mask", "grasp": {"shape": np.zeros((1, 10))},
                       "camera": {"K": np.array([[600, 0, 319], [0, 600, 239], [0, 0, 1]]), "width": 640, "height": 480},
                       "source_mano_side": "left", "canonical_mano_side": "right",
                       "canonicalization": "camera_x_reflection"}

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def test_replaces_geometry_without_touching_inputs(self):
        before = pickle.dumps(self.record)
        result = GeometryOverlay(self.directory, self.dataset).apply("left_sample", self.record)
        self.assertEqual(before, pickle.dumps(self.record))
        self.assertIs(result["image"], self.record["image"])
        self.assertIs(result["depth"], self.record["depth"])
        self.assertNotIn("object_mask", result)
        self.assertEqual(result["mano_parameter_convention"], CONVENTION)
        g = result["grasp"]
        np.testing.assert_allclose(g["landmarks_2d"], np.tile([339, 269], (21, 1)), atol=1e-4)
        np.testing.assert_array_equal(g["t"][0], g["landmarks_3d"][0])
        np.testing.assert_array_equal(g["T_camera_wrist"][:3, 3], g["t"][0])
        np.testing.assert_array_equal(g["T_camera_wrist"][:3, :3], np.eye(3))

    def test_incomplete_overlay_rejected(self):
        self.manifest["status"] = "running"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            GeometryOverlay(self.directory, self.dataset)

    def test_missing_sample_rejected(self):
        overlay = GeometryOverlay(self.directory, self.dataset)
        for missing in ("a_before", "z_after"):
            with self.assertRaises(KeyError):
                overlay.apply(missing, self.record)
            with self.assertRaises(ValueError):
                overlay.validate_samples([missing])
        overlay.validate_samples(["right_sample", "left_sample"])

    def test_wrong_dataset_camera_and_side_rejected(self):
        with self.assertRaisesRegex(ValueError, "different"):
            GeometryOverlay(self.directory, self.root / "other")
        overlay = GeometryOverlay(self.directory, self.dataset)
        bad = dict(self.record, source_mano_side="right")
        with self.assertRaisesRegex(ValueError, "side"):
            overlay.apply("left_sample", bad)
        bad = dict(self.record, camera=dict(self.record["camera"], width=224))
        with self.assertRaisesRegex(ValueError, "camera"):
            overlay.apply("left_sample", bad)

    def test_workers_reopen_readonly_and_return_independent_arrays(self):
        overlay = pickle.loads(pickle.dumps(GeometryOverlay(self.directory, self.dataset)))
        first = overlay.apply("left_sample", self.record)
        first["grasp"]["landmarks_3d"][:] = 99
        second = overlay.apply("left_sample", self.record)
        self.assertLess(float(second["grasp"]["landmarks_3d"].max()), 1)
        self.assertFalse(overlay._arrays["joints"].flags.writeable)

    def test_duplicate_or_unsorted_sample_index_rejected(self):
        np.save(self.directory / "samples.npy", np.array(["same", "same"]))
        with self.assertRaisesRegex(ValueError, "unique"):
            GeometryOverlay(self.directory, self.dataset)

    def test_corrupted_array_shape_rejected(self):
        np.save(self.directory / "params.npy", np.zeros((2, 99), np.float32))
        with self.assertRaisesRegex(ValueError, "params"):
            GeometryOverlay(self.directory, self.dataset)

    def test_corrupted_wrist_rejected(self):
        params = np.load(self.directory / "params.npy")
        params[0, 0] += 1
        np.save(self.directory / "params.npy", params)
        with self.assertRaisesRegex(ValueError, "wrist"):
            GeometryOverlay(self.directory, self.dataset).apply("left_sample", self.record)


if __name__ == "__main__":
    unittest.main()

"""Tests for offline geometry repair; no dataset files are needed."""
import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.repair_shape_gt_geometry import (
    MARKER, atomic_write, collect_files, load_one, params109, repair_batch,
)
from src.models.mano import MANO


class RepairGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.mano = MANO().eval()

    def fixture(self):
        rot = np.eye(3, dtype=np.float32)[:, :2].reshape(-1)
        return {"image": b"unchanged_rgb", "depth": b"unchanged_depth",
                "object_mask": b"unchanged_mask",
                "camera": {"K": np.array([[300, 0, 112], [0, 300, 112], [0, 0, 1]], np.float32)},
                "grasp": {"t": np.array([[0.02, 0.03, 0.6]], np.float32),
                          "R_6d": rot[None], "pose_6d": np.tile(rot, (1, 15, 1)),
                          "shape": np.zeros((1, 10), np.float32),
                          "shape_gt": np.ones((1, 10), np.float32),
                          "landmarks_2d": np.zeros((21, 2), np.float32),
                          "landmarks_3d": np.zeros((21, 3), np.float32),
                          "mesh_vertices": np.zeros((778, 3), np.float32)}}

    def test_target_shape_projection_and_unchanged_inputs(self):
        data = self.fixture()
        original = pickle.dumps(data)
        fixed = repair_batch([data], self.mano, "cpu")[0]
        x = torch.from_numpy(params109(data)[None])
        with torch.no_grad():
            target = self.mano(x, betas=x[:, 99:109])
        j = target["landmarks_3d"][0].numpy() + x[0, :3].numpy()
        projected = j @ data["camera"]["K"].T
        np.testing.assert_allclose(fixed["grasp"]["landmarks_3d"], j, atol=1e-7)
        np.testing.assert_allclose(fixed["grasp"]["landmarks_2d"], projected[:, :2] / projected[:, 2:], atol=3e-5)
        self.assertEqual(pickle.dumps(data), original)
        for key in ("image", "depth", "object_mask"):
            self.assertEqual(fixed[key], data[key])
        for key in ("t", "R_6d", "pose_6d", "shape", "shape_gt"):
            np.testing.assert_array_equal(fixed["grasp"][key], data["grasp"][key])
        self.assertGreater(float(np.abs(fixed["grasp"]["landmarks_2d"]).max()), 0)

    def test_missing_shape_gt_and_invalid_depth_fail(self):
        data = self.fixture()
        del data["grasp"]["shape_gt"]
        with self.assertRaises(KeyError):
            params109(data)
        data = self.fixture()
        data["grasp"]["t"][0, 2] = -1
        with self.assertRaisesRegex(ValueError, "depth"):
            repair_batch([data], self.mano, "cpu")

    def test_repeat_repair_is_idempotent(self):
        first = repair_batch([self.fixture()], self.mano, "cpu")[0]
        second = repair_batch([first], self.mano, "cpu")[0]
        for key in ("landmarks_2d", "landmarks_3d", "mesh_vertices"):
            np.testing.assert_array_equal(first["grasp"][key], second["grasp"][key])

    def test_resume_checks_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "in", root / "out"
            rel = Path("nested/sample.pkl")
            raw = pickle.dumps(self.fixture())
            atomic_write(source / rel, raw)
            _, data, marker = load_one(source, output, rel, "test_signature")
            data[MARKER] = marker
            atomic_write(output / rel, pickle.dumps(data))
            self.assertIsNone(load_one(source, output, rel, "test_signature")[1])
            changed = self.fixture()
            changed["image"] = b"new_source"
            atomic_write(source / rel, pickle.dumps(changed))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_one(source, output, rel, "test_signature")

    def test_sample_lists_deduplicate_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listing = root / "list.txt"
            listing.write_text("nested/a\nnested/a.pkl\nb\n")
            self.assertEqual(len(collect_files(root, [listing], 0)), 2)
            listing.write_text("../outside\n")
            with self.assertRaisesRegex(ValueError, "relative"):
                collect_files(root, [listing], 0)


if __name__ == "__main__":
    unittest.main()

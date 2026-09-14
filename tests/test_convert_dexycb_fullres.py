"""Full-resolution conversion: border retention, reflection and camera consistency."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.convert_dexycb_fullres import (
    mesh_mask, native_inputs, png, source_paths, verify_old_camera,
)


class FullResolutionConversionTest(unittest.TestCase):
    def inputs(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :80] = [12, 34, 56]
        rgb[:, 560:] = [87, 65, 43]
        depth = np.tile(np.arange(640, dtype=np.uint16), (480, 1))
        k = np.array([[600, 0, 310], [0, 602, 245], [0, 0, 1]], dtype=np.float64)
        return rgb, depth, k

    def metadata(self, side="right"):
        return {"source_mano_side": side, "canonical_mano_side": "right",
                "canonicalization": "camera_x_reflection" if side == "left" else "none"}

    def test_right_hand_keeps_both_border_strips_and_depth(self):
        rgb, depth, k = self.inputs()
        image, dep, cam = native_inputs(self.metadata(), rgb, depth, k)
        self.assertEqual(image.shape, (480, 640, 3))
        np.testing.assert_array_equal(image, rgb)
        np.testing.assert_array_equal(dep, depth)
        np.testing.assert_array_equal(cam, k)
        decoded = cv2.imdecode(np.frombuffer(png(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), rgb)
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png(dep), np.uint8), cv2.IMREAD_UNCHANGED), depth)

    def test_left_hand_reflection_preserves_projection_and_pixels(self):
        rgb, depth, k = self.inputs()
        image, dep, cam = native_inputs(self.metadata("left"), rgb, depth, k)
        np.testing.assert_array_equal(image, rgb[:, ::-1])
        np.testing.assert_array_equal(dep, depth[:, ::-1])
        xyz = np.array([0.10, 0.02, 0.6])
        original_uv = k @ xyz
        original_uv = original_uv[:2] / original_uv[2]
        reflected_uv = cam @ (xyz * [-1, 1, 1])
        reflected_uv = reflected_uv[:2] / reflected_uv[2]
        np.testing.assert_allclose(reflected_uv, [639 - original_uv[0], original_uv[1]])

    def test_camera_crosscheck_rejects_wrong_serial_intrinsics(self):
        _, _, k = self.inputs()
        old = k.copy()
        old[0, 2] -= 80
        old[:2] *= 224 / 480
        record = {"camera": {"K": old, "width": 224, "height": 224}}
        verify_old_camera(record, k, 640, 480)
        bad = k.copy()
        bad[0, 0] += 20
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_old_camera(record, bad, 640, 480)

    def test_mask_keeps_triangles_crossing_the_border(self):
        # 所有顶点均在画布外，但三角形覆盖画布；不能先丢弃越界顶点。
        verts = np.array([[-100, -100, 1], [100, -100, 1], [0, 100, 1]], np.float32)
        mask = mesh_mask(verts, np.array([[0, 1, 2]]), np.eye(3), 20, 10)
        self.assertEqual(mask.shape, (10, 20))
        self.assertTrue((mask == 255).all())

    def test_invalid_canonicalization_is_rejected(self):
        rgb, depth, k = self.inputs()
        record = self.metadata("left")
        record["canonicalization"] = "none"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            native_inputs(record, rgb, depth, k)

    def test_source_path_mapping_and_invalid_stem(self):
        root = Path("/dataset")
        rgb, depth, _ = source_paths(root, "dexycb_20200709-subject-01_20200709_141754_836212060125_000009")
        self.assertEqual(rgb, root / "20200709-subject-01/20200709_141754/836212060125/color_000009.jpg")
        self.assertEqual(depth.name, "aligned_depth_to_color_000009.png")
        with self.assertRaises(ValueError):
            source_paths(root, "../bad")


if __name__ == "__main__":
    unittest.main()

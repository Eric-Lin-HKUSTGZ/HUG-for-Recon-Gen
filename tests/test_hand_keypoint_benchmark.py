import unittest

import numpy as np

from src.benchmark_hand_keypoints import (
    bbox_iou,
    crop_affine,
    evaluate_method,
    expanded_square_bbox,
    invert_affine,
    transform_points,
)


class HandKeypointBenchmarkTest(unittest.TestCase):
    def test_expanded_square_bbox(self):
        actual = expanded_square_bbox(np.array([10, 20, 30, 60]), expand=1.5)
        np.testing.assert_allclose(actual, [-10, 10, 50, 70])

    def test_affine_round_trip(self):
        points = np.array([[1.0, 2.0], [30.0, 40.0]], np.float32)
        affine = crop_affine(np.array([-10, 10, 50, 70]), 224)
        recovered = transform_points(transform_points(points, affine), invert_affine(affine))
        np.testing.assert_allclose(recovered, points, atol=1e-5)

    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])), 1 / 3)

    def test_misses_reduce_full_pck_but_not_conditional_pck(self):
        gt = np.stack([np.arange(21), np.zeros(21)], axis=1).astype(float)
        manifests = [
            {"sample": "a", "gt_keypoints_xy": gt.tolist(), "gt_valid": [True] * 21},
            {"sample": "b", "gt_keypoints_xy": gt.tolist(), "gt_valid": [True] * 21},
        ]
        predictions = [
            {"sample": "a", "returned": True, "keypoints_xy": gt.tolist()},
            {"sample": "b", "returned": False},
        ]
        metrics, _, _ = evaluate_method(manifests, predictions)
        self.assertEqual(metrics["return_rate"], 0.5)
        self.assertEqual(metrics["pck_0.20_conditional"], 1.0)
        self.assertEqual(metrics["pck_0.20_all"], 0.5)

    def test_frame_with_fewer_than_two_visible_gt_joints_is_not_evaluable(self):
        gt = np.stack([np.arange(21), np.zeros(21)], axis=1).astype(float)
        manifests = [
            {"sample": "a", "gt_keypoints_xy": gt.tolist(), "gt_valid": [True] + [False] * 20}
        ]
        predictions = [{"sample": "a", "returned": True, "keypoints_xy": gt.tolist()}]
        metrics, _, _ = evaluate_method(manifests, predictions)
        self.assertEqual(metrics["valid_samples"], 0)
        self.assertEqual(metrics["returned_samples"], 0)


if __name__ == "__main__":
    unittest.main()

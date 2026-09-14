"""Regression tests for Oracle A GT-keypoint conditioning."""

import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.dataloader.grasp_dataset import GraspDataset


class OracleAKeypointsTest(unittest.TestCase):
    @staticmethod
    def _dataset(source="gt"):
        dataset = object.__new__(GraspDataset)
        dataset.image_size = 224
        dataset.keypoint_source = source
        return dataset

    def test_gt_keypoints_are_transformed_for_rgb_but_preserved_for_depth(self):
        dataset = self._dataset()
        points = np.stack(
            [np.linspace(40.0, 180.0, 21), np.linspace(30.0, 170.0, 21)],
            axis=1,
        ).astype(np.float32)
        affine = dataset._crop_affine(
            np.array([30.0, 20.0, 190.0, 180.0], dtype=np.float32), 224
        )
        rgb_original = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb_crop = np.zeros((224, 224, 3), dtype=np.uint8)

        crop, crop_valid, original, depth_valid = dataset._hand_keypoints(
            {"grasp": {"landmarks_2d": points}},
            rgb_crop,
            rgb_original,
            affine,
        )

        np.testing.assert_allclose(crop, dataset._transform_points(points, affine))
        np.testing.assert_allclose(original, points)
        self.assertTrue(crop_valid.all())
        self.assertTrue(depth_valid.all())

    def test_finite_out_of_frame_gt_remains_valid_in_expanded_train_crop(self):
        dataset = self._dataset()
        points = np.full((21, 2), [10.0, 100.0], dtype=np.float32)
        points[0] = [-15.0, 100.0]
        bbox = dataset._expanded_square_bbox(
            dataset._bbox_from_keypoints(points), 1.5
        )
        affine = dataset._crop_affine(bbox, 224)

        _, crop_valid, _, depth_valid = dataset._hand_keypoints(
            {"grasp": {"landmarks_2d": points}},
            np.zeros((224, 224, 3), dtype=np.uint8),
            np.zeros((224, 224, 3), dtype=np.uint8),
            affine,
        )

        self.assertTrue(crop_valid.all())
        self.assertTrue(depth_valid.all())

    def test_detector_crop_exclusion_does_not_remove_gt_from_depth_roi(self):
        dataset = self._dataset()
        points = np.full((21, 2), [100.0, 100.0], dtype=np.float32)
        points[0] = [250.0, 100.0]
        affine = dataset._crop_affine(
            np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float32), 224
        )

        _, crop_valid, original, depth_valid = dataset._hand_keypoints(
            {"grasp": {"landmarks_2d": points}},
            np.zeros((224, 224, 3), dtype=np.uint8),
            np.zeros((240, 320, 3), dtype=np.uint8),
            affine,
        )

        self.assertFalse(bool(crop_valid[0]))
        self.assertTrue(bool(depth_valid[0]))
        np.testing.assert_allclose(original[0], points[0])

    def test_gt_mode_fails_loudly_without_reconstruction_labels(self):
        dataset = self._dataset()
        with self.assertRaisesRegex(ValueError, "requires grasp.landmarks_2d"):
            dataset._hand_keypoints(
                {},
                np.zeros((224, 224, 3), dtype=np.uint8),
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.eye(3, dtype=np.float32)[:2],
            )

    def test_oracle_config_preserves_v23_crop_policy(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "configs" / "train_handrecon.yaml")
        hand_crop = cfg.trainer.data.hand_crop

        self.assertTrue(bool(hand_crop.enabled))
        self.assertEqual(hand_crop.keypoint_source, "gt")
        self.assertAlmostEqual(float(hand_crop.expand), 1.5)
        self.assertAlmostEqual(float(hand_crop.depth_radius_expand), 1.25)
        self.assertEqual(
            hand_crop.detector_weights,
            "/root/code/vepfs/GPGFormer/weights/detector.pt",
        )
        self.assertTrue(bool(cfg.trainer.model.use_skeleton_condition))
        self.assertFalse(bool(cfg.trainer.model.use_query_condition))
        self.assertEqual(cfg.trainer.model.query_fusion_mode, "legacy_broadcast")


if __name__ == "__main__":
    unittest.main()

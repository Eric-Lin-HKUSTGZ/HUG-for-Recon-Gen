import unittest

import numpy as np

from src.dataloader.augmented_grasp_dataset import AugmentedGraspDataset


class AugmentedConditionCacheTest(unittest.TestCase):
    @staticmethod
    def make_dataset() -> AugmentedGraspDataset:
        dataset = AugmentedGraspDataset.__new__(AugmentedGraspDataset)
        keypoints = np.tile(np.array([[10.0, 20.0]], np.float32), (21, 1))
        keypoints[-1] = [200.0, 300.0]
        dataset._condition_cache = {
            "detector_bbox_xyxy": np.array([[10.0, 20.0, 30.0, 40.0]], np.float32),
            "crop_bbox_xyxy": np.array([[5.0, 15.0, 35.0, 45.0]], np.float32),
            "detector_hit": np.array([True]),
            "keypoints_xy": keypoints[None],
            "keypoint_scores": np.ones((1, 21), np.float32),
            "pose_returned": np.array([True]),
        }
        dataset._condition_cache_index = None
        dataset.hand_crop_expand = 1.5
        return dataset

    def test_cached_condition_follows_applied_affine(self):
        dataset = self.make_dataset()
        dataset._augmentation_state = {
            "applied_affine_matrix": np.array(
                [[0.0, -1.0, 100.0], [1.0, 0.0, 5.0]], np.float32
            ),
            "affine_output_shape": (100, 100),
        }

        condition = dataset._condition_for_sample(0)

        np.testing.assert_allclose(
            condition["detector_bbox_xyxy"], [60.0, 15.0, 80.0, 35.0]
        )
        np.testing.assert_allclose(
            condition["crop_bbox_xyxy"], [52.0, 7.0, 88.0, 43.0]
        )
        np.testing.assert_allclose(condition["keypoints_xy"][0], [80.0, 15.0])
        self.assertEqual(float(condition["keypoint_scores"][0]), 1.0)
        self.assertEqual(float(condition["keypoint_scores"][-1]), 0.0)

        # The immutable cache is reused by other samples and workers.
        np.testing.assert_allclose(
            dataset._condition_cache["detector_bbox_xyxy"][0],
            [10.0, 20.0, 30.0, 40.0],
        )
        np.testing.assert_allclose(
            dataset._condition_cache["keypoints_xy"][0, 0], [10.0, 20.0]
        )

    def test_rejected_affine_does_not_transform_cached_condition(self):
        dataset = self.make_dataset()
        stale_matrix = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 9.0]], np.float32)
        dataset._augmentation_state = {
            "applied_affine_matrix": stale_matrix,
            "affine_output_shape": (32, 32),
        }
        dataset.affine_cfg = {"min_visible_landmarks": 0.95}
        grasp_data = {
            "image": dataset._encode_rgb(np.zeros((32, 32, 3), np.uint8)),
            "condition_point": [1000.0, 1000.0],
        }
        params = {"scale": 1.0, "angle_deg": 0.0, "tx_frac": 0.0, "ty_frac": 0.0}

        result = dataset._apply_affine_to_grasp_data(grasp_data, params)
        condition = dataset._condition_for_sample(0)

        self.assertIs(result, grasp_data)
        self.assertIsNone(dataset._augmentation_state["applied_affine_matrix"])
        self.assertIsNone(dataset._augmentation_state["affine_output_shape"])
        np.testing.assert_allclose(
            condition["detector_bbox_xyxy"], [10.0, 20.0, 30.0, 40.0]
        )
        np.testing.assert_allclose(condition["keypoints_xy"][0], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()

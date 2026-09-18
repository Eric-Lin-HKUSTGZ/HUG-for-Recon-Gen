import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.dataloader.grasp_dataset import GraspDataset
from src.eval_test import DistributedEvalSampler


class _Instances:
    def __init__(self, keypoints, scores):
        self.keypoints = keypoints[None]
        self.keypoint_scores = scores[None]


class _Result:
    def __init__(self, keypoints, scores):
        self.pred_instances = _Instances(keypoints, scores)


class _Model:
    dataset_meta = {}

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def test_step(self, _data):
        self.calls += 1
        return [self.result]


class RTMPoseDatasetTest(unittest.TestCase):
    def make_dataset(self, scores):
        dataset = GraspDataset.__new__(GraspDataset)
        dataset.image_size = 224
        dataset.rtmpose_min_joint_conf = 0.1
        dataset.rtmpose_min_mean_conf = 0.0
        keypoints = np.stack(
            [np.arange(21, dtype=np.float32), np.arange(21, dtype=np.float32)],
            axis=1,
        )
        model = _Model(_Result(keypoints, np.asarray(scores, dtype=np.float32)))
        dataset._rtmpose_runtime = (model, lambda value: value, lambda value: value)
        return dataset, model, keypoints

    def test_detector_miss_skips_top_down_pose(self):
        dataset, model, _ = self.make_dataset(np.ones(21))
        outputs = dataset._rtmpose_keypoints(
            np.zeros((32, 32, 3), np.uint8),
            np.array([0, 0, 31, 31], np.float32),
            np.eye(3, dtype=np.float32)[:2],
            detector_hit=False,
        )
        self.assertEqual(model.calls, 0)
        self.assertFalse(outputs[1].any())
        self.assertFalse(outputs[3].any())

    def test_scores_mask_joints_and_affine_maps_crop_coordinates(self):
        scores = np.ones(21, dtype=np.float32)
        scores[5] = 0.05
        dataset, model, keypoints = self.make_dataset(scores)
        affine = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0]], np.float32)
        crop_xy, crop_valid, confidence, original_xy, original_valid = (
            dataset._rtmpose_keypoints(
                np.zeros((32, 32, 3), np.uint8),
                np.array([0, 0, 31, 31], np.float32),
                affine,
                detector_hit=True,
            )
        )
        self.assertEqual(model.calls, 1)
        np.testing.assert_allclose(original_xy, keypoints)
        np.testing.assert_allclose(crop_xy, keypoints * 2 + [3, 4])
        np.testing.assert_allclose(confidence, scores)
        self.assertFalse(original_valid[5])
        self.assertFalse(crop_valid[5])
        self.assertEqual(int(original_valid.sum()), 20)

    def test_distributed_eval_sampler_has_no_padding_duplicates(self):
        shards = [
            list(DistributedEvalSampler(range(10), num_replicas=3, rank=rank))
            for rank in range(3)
        ]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_full_frame_fallback_keeps_fixed_model_input_size(self):
        dataset = GraspDataset.__new__(GraspDataset)
        dataset.image_size = 224
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        camera_k = np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        crop, crop_k, affine = dataset._crop_rgb(
            rgb,
            camera_k,
            np.array([0.0, 0.0, 639.0, 479.0], dtype=np.float32),
        )
        self.assertEqual(crop.shape, (224, 224, 3))
        np.testing.assert_allclose(crop_k, np.vstack([affine, [0, 0, 1]]) @ camera_k)

    def test_detector_auto_uses_current_ddp_gpu(self):
        calls = []

        def detector(_image, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

        dataset = GraspDataset.__new__(GraspDataset)
        dataset._hand_detector = detector
        dataset.detector_device = "auto"
        dataset.detector_conf = 0.25
        dataset.detector_iou = 0.7
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.current_device", return_value=3
        ):
            self.assertIsNone(
                dataset._detect_hand_bbox(np.zeros((32, 32, 3), np.uint8))
            )
        self.assertEqual(calls[0]["device"], 3)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the HUG Eq. 1 training objective."""

import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.train import compute_loss


def _loss_dicts(
    velocity_pred,
    velocity_target,
    pred_joints,
    target_joints,
    pred_vertices=None,
    target_vertices=None,
):
    batch_size = velocity_pred.shape[0]
    if pred_vertices is None:
        pred_vertices = torch.zeros(batch_size, 8, 3)
    if target_vertices is None:
        target_vertices = torch.zeros_like(pred_vertices)
    return (
        {
            "params_norm": velocity_pred,
            "landmarks_3d": pred_joints,
            "vertices": pred_vertices,
        },
        {
            "params_norm": velocity_target,
            "landmarks_3d": target_joints,
            "vertices": target_vertices,
        },
    )


class OriginalHugLossTest(unittest.TestCase):
    def test_flow_is_one_uniform_mse_over_the_complete_109d_state(self):
        velocity_pred = torch.ones(2, 109)
        velocity_target = torch.zeros_like(velocity_pred)
        joints = torch.zeros(2, 21, 3)
        preds, targets = _loss_dicts(
            velocity_pred, velocity_target, joints, joints.clone()
        )

        loss, comps = compute_loss(
            preds,
            targets,
            time_weight=torch.ones(2),
            lambda_v=1.0,
            lambda_3d=0.0,
            lambda_mesh_3d=0.0,
        )

        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        self.assertAlmostEqual(comps["lv"], 1.0, places=6)
        self.assertAlmostEqual(comps["loss_flow_weighted"], 1.0, places=6)

    def test_3d_is_camera_frame_landmark_l1_weighted_by_one_minus_t(self):
        velocity = torch.zeros(2, 109)
        target_joints = torch.zeros(2, 21, 3)
        pred_joints = target_joints.clone()
        pred_joints[0] = torch.tensor([1.0, 2.0, 3.0])
        pred_joints[1] = torch.tensor([-2.0, 0.0, 1.0])
        preds, targets = _loss_dicts(
            velocity, velocity.clone(), pred_joints, target_joints
        )

        loss, comps = compute_loss(
            preds,
            targets,
            time_weight=torch.tensor([1.0, 0.25]),
            lambda_v=0.0,
            lambda_3d=20.0,
            lambda_mesh_3d=0.0,
        )

        expected_raw_l1 = (2.0 + 1.0) / 2.0
        expected_time_weighted = (1.0 * 2.0 + 0.25 * 1.0) / 2.0
        self.assertAlmostEqual(comps["l3d"], expected_raw_l1, places=6)
        self.assertAlmostEqual(
            comps["l3d_time_weighted"], expected_time_weighted, places=6
        )
        self.assertAlmostEqual(
            comps["loss_3d_weighted"], 20.0 * expected_time_weighted, places=6
        )
        self.assertAlmostEqual(loss.item(), 20.0 * expected_time_weighted, places=6)

    def test_mesh_is_camera_frame_vertex_l1_weighted_by_one_minus_t(self):
        velocity = torch.zeros(2, 109)
        joints = torch.zeros(2, 21, 3)
        target_vertices = torch.zeros(2, 8, 3)
        pred_vertices = target_vertices.clone()
        pred_vertices[0] = torch.tensor([3.0, 0.0, 0.0])
        pred_vertices[1] = torch.tensor([0.0, -3.0, 3.0])
        preds, targets = _loss_dicts(
            velocity,
            velocity.clone(),
            joints,
            joints.clone(),
            pred_vertices,
            target_vertices,
        )

        loss, comps = compute_loss(
            preds,
            targets,
            time_weight=torch.tensor([0.5, 1.0]),
            lambda_v=0.0,
            lambda_3d=0.0,
            lambda_mesh_3d=10.0,
        )

        expected_raw_l1 = (1.0 + 2.0) / 2.0
        expected_time_weighted = (0.5 * 1.0 + 1.0 * 2.0) / 2.0
        self.assertAlmostEqual(comps["lmesh"], expected_raw_l1, places=6)
        self.assertAlmostEqual(
            comps["lmesh_time_weighted"], expected_time_weighted, places=6
        )
        self.assertAlmostEqual(loss.item(), 10.0 * expected_time_weighted, places=6)

    def test_only_flow_joint_and_mesh_losses_contribute_gradients(self):
        torch.manual_seed(7)
        velocity_pred = torch.randn(2, 109, requires_grad=True)
        velocity_target = torch.zeros_like(velocity_pred)
        pred_joints = torch.randn(2, 21, 3, requires_grad=True)
        target_joints = torch.zeros_like(pred_joints)
        pred_vertices = torch.randn(2, 8, 3, requires_grad=True)
        target_vertices = torch.zeros_like(pred_vertices)
        preds, targets = _loss_dicts(
            velocity_pred,
            velocity_target,
            pred_joints,
            target_joints,
            pred_vertices,
            target_vertices,
        )

        loss, comps = compute_loss(
            preds,
            targets,
            time_weight=torch.tensor([0.8, 0.4]),
            lambda_v=1.0,
            lambda_3d=20.0,
            lambda_mesh_3d=10.0,
        )
        loss.backward()

        self.assertGreater(velocity_pred.grad.abs().sum().item(), 0.0)
        self.assertGreater(pred_joints.grad.abs().sum().item(), 0.0)
        self.assertGreater(pred_vertices.grad.abs().sum().item(), 0.0)
        self.assertEqual(
            set(comps),
            {
                "loss",
                "lv",
                "l3d",
                "l3d_time_weighted",
                "lmesh",
                "lmesh_time_weighted",
                "loss_flow_weighted",
                "loss_3d_weighted",
                "loss_mesh_3d_weighted",
                "time_weight_mean",
                "train_mpjpe_mm",
                "train_mpvpe_mm",
            },
        )

    def test_training_config_restores_legacy_query_and_disables_extensions(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "configs" / "train_handrecon.yaml")
        self.assertEqual(
            cfg.trainer.model.query_fusion_mode,
            "legacy_broadcast",
        )
        self.assertEqual(float(cfg.trainer.train.lambda_v), 1.0)
        self.assertEqual(float(cfg.trainer.train.lambda_3d), 20.0)
        self.assertEqual(float(cfg.trainer.train.lambda_mesh_3d), 10.0)
        inactive = (
            "lambda_joint_local",
            "lambda_vertex_local",
            "lambda_theta",
            "lambda_bone_direction",
            "lambda_bone_ratio",
            "lambda_shape_x0",
            "lambda_2d",
            "lambda_translation",
            "lambda_rotation",
        )
        for key in inactive:
            self.assertEqual(float(cfg.trainer.train[key]), 0.0, key)


if __name__ == "__main__":
    unittest.main()

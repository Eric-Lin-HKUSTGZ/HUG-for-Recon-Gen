"""MANO geometry and gradient checks. Requires repository MANO assets."""
import copy
import os
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.models.native_mano import NativeCanonicalMANO


class NativeMANOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = NativeCanonicalMANO(os.environ.get(
            "DEXYCB_LEFT_ASSET", "/root/code/vepfs/GPGFormer/weights/mano/MANO_LEFT.pkl")).eval()

    def fixture(self):
        rng = np.random.default_rng(1234)
        aa = rng.normal(size=(4, 16, 3)).astype(np.float32) * 0.25
        aa[0, 0] = [np.pi - 1e-6, 0, 0]
        beta = torch.tensor(rng.normal(size=(4, 10)), dtype=torch.float32)
        left = torch.tensor([True, False, True, False])
        canonical_aa = aa.copy()
        canonical_aa[left.numpy()] *= np.array([1, -1, -1], np.float32)
        rots = Rotation.from_rotvec(canonical_aa.reshape(-1, 3)).as_matrix().reshape(4, 16, 3, 3)
        x = torch.cat([torch.tensor([[0.02, 0.01, 0.7]]).repeat(4, 1),
                       torch.tensor(rots[..., :2].reshape(4, 96), dtype=torch.float32), beta], dim=1)
        return x, left, aa, beta

    def test_mixed_sides_match_native_forward_with_mirror(self):
        x, left, aa, beta = self.fixture()
        actual = self.model(x, source_is_left=left)
        for i in range(4):
            layer = self.model.left_layer if left[i] else self.model.mano_layer
            expected = layer(torch.tensor(aa[i].reshape(1, 48)), beta[i:i+1])
            reflection = torch.tensor([-1, 1, 1]) if left[i] else torch.ones(3)
            torch.testing.assert_close(actual["landmarks_3d"][i], expected.joints[0] * reflection, atol=2e-6, rtol=1e-4)
            torch.testing.assert_close(actual["vertices"][i], expected.verts[0] * reflection, atol=2e-6, rtol=1e-4)

    def test_gradient_reaches_pose_and_shape_on_both_sides(self):
        x, left, _, _ = self.fixture()
        x.requires_grad_()
        result = self.model(x, source_is_left=left)
        loss = result["landmarks_3d"].square().sum() + result["vertices"].square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[:, 3:99].abs().sum(1) > 0).all())
        self.assertTrue((x.grad[:, 99:].abs().sum(1) > 0).all())

    def test_empty_side_branch_and_missing_metadata(self):
        x, left, _, _ = self.fixture()
        for mask in (left, ~left):
            out = self.model(x[mask], source_is_left=left[mask])
            self.assertEqual(out["vertices"].shape, (2, 778, 3))
        with self.assertRaisesRegex(ValueError, "source_is_left"):
            self.model(x)

    def test_targets_are_supplied_gt_not_regenerated(self):
        # Construct only the geometry portion; no encoder downloads or GPU.
        from src.models.grasp_model import GraspFlowModel
        model = GraspFlowModel.__new__(GraspFlowModel)
        torch.nn.Module.__init__(model)
        model.mano_geometry = "native_side_v1"
        model.mano = self.model
        x, left, _, _ = self.fixture()
        joints = torch.full((4, 21, 3), 123.0)
        mesh = torch.full((4, 778, 3), 456.0)
        pred, target = model._build_dicts(x, x, x, x, source_is_left=left,
                                         gt_joints_3d=joints, gt_vertices=mesh)
        self.assertIs(target["landmarks_3d"], joints)
        self.assertIs(target["vertices"], mesh)
        with self.assertRaisesRegex(ValueError, "converted native GT"):
            model._build_dicts(x, x, x, x)

    def test_animation_uses_native_side_and_ends_at_prediction(self):
        from src.models.mano import mano_params_to_animation
        x, left, _, beta = self.fixture()
        for i in (0, 1):
            vertices, joints = mano_params_to_animation(
                x[i], beta[i], self.model, n_frames=3,
                source_is_left=bool(left[i]),
            )
            expected = self.model(x[i:i+1], source_is_left=left[i:i+1])
            torch.testing.assert_close(
                torch.from_numpy(vertices[-1]),
                expected["vertices"][0] + x[i, :3], atol=2e-6, rtol=1e-4,
            )
            torch.testing.assert_close(
                torch.from_numpy(joints[-1]),
                expected["landmarks_3d"][0] + x[i, :3], atol=2e-6, rtol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()

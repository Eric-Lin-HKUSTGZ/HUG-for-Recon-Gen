"""Regression tests for query-conditioned patch fusion."""

import unittest

import torch

from src.models.fusion import PatchFusion


class QueryFusionTest(unittest.TestCase):
    def _inputs(self, batch_size=2):
        torch.manual_seed(7)
        rgb = torch.randn(batch_size, 4, 8)
        depth = torch.randn(batch_size, 4, 6)
        centroids = torch.tensor(
            [
                [-0.10, -0.10, 1.0],
                [0.10, -0.10, 1.0],
                [-0.10, 0.10, 1.0],
                [0.10, 0.10, 1.0],
            ]
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        camera_k = torch.tensor(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        point = torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch_size, 1)
        return point, rgb, depth, centroids, camera_k

    def _model(self, mode):
        torch.manual_seed(11)
        return PatchFusion(
            d_rgb_patch=8,
            d_depth_patch=6,
            d_model=32,
            n_patches=4,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            patch_grid_size=2,
            use_rgb=True,
            use_depth=True,
            use_pointpainting=True,
            image_size=4,
            query_fusion_mode=mode,
        )

    def test_query_to_scene_shape_and_query_dependence(self):
        model = self._model("query_to_scene").eval()
        point, rgb, depth, centroids, camera_k = self._inputs()
        out = model(point, rgb, depth, centroids, camera_k)
        moved_point = point.clone()
        moved_point[:, 0] += 0.08
        moved = model(moved_point, rgb, depth, centroids, camera_k)

        self.assertEqual(tuple(out.shape), (2, 4, 32))
        self.assertGreater((out - moved).abs().max().item(), 1e-6)

    def test_query_and_key_attention_paths_receive_gradients(self):
        model = self._model("query_to_scene").train()
        point, rgb, depth, centroids, camera_k = self._inputs()
        out = model(point, rgb, depth, centroids, camera_k)
        weights = torch.linspace(-1.0, 1.0, out.numel()).reshape_as(out)
        (out * weights).sum().backward()

        attn = model.query_to_scene_attn.cross_attn
        q_grad = attn.q_proj.weight.grad
        key_grad = attn.kv_proj.weight.grad[: model.d_model]
        rel_grad = model.relative_pos_embed_3d.raw_proj.weight.grad
        point_grad = model.point_proj[0].weight.grad
        for name, grad in (
            ("query", q_grad),
            ("key", key_grad),
            ("relative_xyz", rel_grad),
            ("point", point_grad),
        ):
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0.0, name)

    def test_legacy_mode_keeps_original_module_and_shape(self):
        model = self._model("legacy_broadcast").eval()
        point, rgb, depth, centroids, camera_k = self._inputs()
        out = model(point, rgb, depth, centroids, camera_k)

        self.assertEqual(tuple(out.shape), (2, 4, 32))
        self.assertTrue(hasattr(model, "point_cross_attn"))
        self.assertFalse(hasattr(model, "query_to_scene_attn"))

    def test_2d_query_mode_does_not_create_unused_3d_relative_branch(self):
        model = PatchFusion(
            d_rgb_patch=8,
            d_depth_patch=6,
            d_model=32,
            n_patches=4,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            patch_grid_size=2,
            use_rgb=True,
            use_depth=True,
            use_pointpainting=True,
            image_size=4,
            use_2d_point=True,
            query_fusion_mode="query_to_scene",
        )
        _, rgb, depth, centroids, camera_k = self._inputs()
        point_2d = torch.tensor([[0.5, 0.5], [0.4, 0.6]])
        out = model(point_2d, rgb, depth, centroids, camera_k)
        out.square().mean().backward()

        self.assertEqual(tuple(out.shape), (2, 4, 32))
        self.assertFalse(hasattr(model, "relative_pos_embed_3d"))
        self.assertIsNotNone(
            model.query_to_scene_attn.cross_attn.q_proj.weight.grad
        )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "query_fusion_mode"):
            self._model("invalid")


if __name__ == "__main__":
    unittest.main()

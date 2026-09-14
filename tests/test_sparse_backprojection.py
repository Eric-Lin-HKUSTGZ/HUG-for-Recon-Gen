"""Sparse pixel sampling must preserve the legacy point-cloud distribution."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.utils import pcl_utils as pcl


def legacy(depth, rgb, K, n_points, rng=None, **kwargs):
    xyz, colors = pcl.backproject_to_pcl(depth, rgb, K, **kwargs)
    xyz, colors = pcl.sample_fixed_n(xyz, colors, n_points, rng=rng)
    return torch.from_numpy(xyz).float(), torch.from_numpy(colors).float()/255.0


class SparseBackprojectionTest(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(42)
        depth = rng.uniform(0.2, 2.0, (17, 23)).astype(np.float32)
        depth.flat[:6] = [0, -1, np.nan, np.inf, 3.0, 4.0]
        depth[3:8, 5:11] = 0
        rgb = rng.integers(0, 256, (*depth.shape, 3), dtype=np.uint8)
        # Includes affine rotation/skew and a nontrivial full camera matrix.
        K = np.array([[250., -23., 11.], [21., 245., 8.], [0., 0., 1.]], np.float32)
        return depth, rgb, K

    def assert_pair(self, a, b):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=1e-7)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)

    def test_generator_and_global_rng_match_dense_reference(self):
        depth,rgb,K = self.inputs()
        count = int(((depth>0)&(depth<3)).sum())
        for n in (0, 8, count, 4096):
            for seed in (0, 11, 72):
                with self.subTest(n=n,seed=seed):
                    a_rng,b_rng=np.random.default_rng(seed),np.random.default_rng(seed)
                    a=legacy(depth,rgb,K,n,rng=a_rng)
                    b=pcl.depth_to_pcl_tensors(depth,rgb,K,n_points=n,rng=b_rng)
                    self.assert_pair(a,b)
                    np.testing.assert_array_equal(a_rng.random(8),b_rng.random(8))
                    np.random.seed(seed)
                    a=legacy(depth,rgb,K,n);next_a=np.random.rand(8)
                    np.random.seed(seed)
                    b=pcl.depth_to_pcl_tensors(depth,rgb,K,n_points=n);next_b=np.random.rand(8)
                    self.assert_pair(a,b)
                    np.testing.assert_array_equal(next_a,next_b)

    def test_sphere_and_partial_sphere_arguments_preserve_reference(self):
        depth,rgb,K=self.inputs()
        for kwargs in ({'center':np.array([0.,0.,1.]),'crop_radius':0.3},
                       {'center':np.array([0.,0.,1.]),'crop_radius':0.0},
                       {'center':np.array([0.,0.,1.])}, {'crop_radius':0.3}):
            with self.subTest(kwargs=kwargs):
                a=legacy(depth,rgb,K,4096,rng=np.random.default_rng(3),**kwargs)
                b=pcl.depth_to_pcl_tensors(depth,rgb,K,rng=np.random.default_rng(3),**kwargs)
                self.assert_pair(a,b)

    def test_empty_valid_set_returns_zeros_without_consuming_rng(self):
        _,rgb,K=self.inputs();depth=np.zeros(rgb.shape[:2],np.float32)
        rng,ref=np.random.default_rng(9),np.random.default_rng(9)
        xyz,colors=pcl.depth_to_pcl_tensors(depth,rgb,K,rng=rng)
        self.assertEqual(tuple(xyz.shape),(4096,3))
        self.assertEqual(torch.count_nonzero(xyz).item(),0)
        self.assertEqual(torch.count_nonzero(colors).item(),0)
        np.testing.assert_array_equal(rng.random(8),ref.random(8))

    def test_only_sampled_pixels_are_backprojected_and_tensors_work(self):
        depth=np.ones((480,640),np.float32)
        rgb=np.zeros((480,640,3),np.uint8);K=np.eye(3,dtype=np.float32)
        with patch.object(pcl,'backproject_depth_np',side_effect=AssertionError('dense projection used')):
            with patch.object(pcl,'backproject_pixels_np',wraps=pcl.backproject_pixels_np) as project:
                xyz,_=pcl.depth_to_pcl_tensors(torch.from_numpy(depth),torch.from_numpy(rgb),
                                              torch.from_numpy(K),rng=np.random.default_rng(2))
                self.assertEqual(tuple(xyz.shape),(4096,3))
                self.assertEqual(project.call_args.args[0].shape,(4096,3))
                self.assertEqual(project.call_count,1)


if __name__ == '__main__':
    unittest.main()

"""Check the existing configurable fusion paths used by v26 and v27."""
import unittest
import copy
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.models.fusion import PatchFusion
from src.models.grasp_model import GraspFlowModel


class _DepthStub(torch.nn.Module):
    def forward(self, xyz, rgb_pcl=None):
        return rgb_pcl, xyz


class RGBDepthTokensTest(unittest.TestCase):
    def test_both_modes_forward_backward_and_modality_dependence(self):
        torch.set_num_threads(1)
        for painting in (True, False):
            with self.subTest(use_pointpainting=painting):
                torch.manual_seed(17)
                model = PatchFusion(
                    d_rgb_patch=8, d_depth_patch=6, d_model=32,
                    n_patches=4, n_layers=1, n_heads=4, dropout=0.0,
                    patch_grid_size=2, image_size=28,
                    use_rgb=True, use_depth=True, use_pointpainting=painting,
                    use_query_condition=False, use_skeleton_condition=True,
                ).eval()
                rgb = torch.randn(2, 4, 8, requires_grad=True)
                depth = torch.randn(2, 4, 6, requires_grad=True)
                xyz = torch.tensor([[[-0.1,-0.1,1.],[0.1,-0.1,1.],
                                     [-0.1,0.1,1.],[0.1,0.1,1.]]]).repeat(2,1,1)
                K = torch.tensor([[[50.,0.,14.],[0.,50.,14.],[0.,0.,1.]]]).repeat(2,1,1)
                kwargs = dict(point=torch.zeros(2,3), depth_centroids=xyz,
                              camera_K=K, hand_keypoints_2d=torch.ones(2,21,2)*14,
                              hand_keypoints_valid=torch.ones(2,21,dtype=torch.bool))
                out = model(rgb_patches=rgb, depth_patches=depth, **kwargs)
                self.assertEqual(tuple(out.shape), (2, 4 if painting else 8, 32))
                self.assertTrue(torch.isfinite(out).all())
                for changed_rgb, changed_depth in ((rgb+0.5,depth),(rgb,depth+0.5)):
                    changed = model(rgb_patches=changed_rgb,depth_patches=changed_depth,**kwargs)
                    self.assertGreater((out-changed).abs().max().item(),1e-6)
                out.square().mean().backward()
                for grad in (rgb.grad,depth.grad):
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(),0)
                # DDP without find_unused_parameters must see all trainable parameters.
                self.assertEqual([n for n,p in model.named_parameters()
                                  if p.requires_grad and p.grad is None], [])

    def test_checkpointing_preserves_dropout_outputs_and_gradients(self):
        torch.set_num_threads(1)
        model = GraspFlowModel.__new__(GraspFlowModel)
        torch.nn.Module.__init__(model)
        model.use_2d_point = True
        model.use_rgb = model.use_depth = model.pcl_use_rgb = True
        model.image_encoder = torch.nn.Identity()
        model.depth_encoder = _DepthStub()
        model.fusion = PatchFusion(
            d_rgb_patch=8, d_depth_patch=6, d_model=32,
            n_patches=4, n_layers=1, n_heads=4, dropout=0.1,
            patch_grid_size=2, image_size=28,
            use_pointpainting=False, use_query_condition=False,
            use_skeleton_condition=True,
        )
        model.train()
        other = copy.deepcopy(model)
        other.fusion.activation_checkpointing = True
        kwargs = dict(
            point_uv=torch.ones(2,3), camera_K=torch.eye(3).repeat(2,1,1),
            rgb=torch.randn(2,4,8), pcl_xyz=torch.randn(2,4,3),
            pcl_rgb=torch.randn(2,4,6), hand_keypoints_2d=torch.ones(2,21,2),
            hand_keypoints_valid=torch.ones(2,21,dtype=torch.bool),
        )
        torch.manual_seed(31)
        a = model.encode_scene(**kwargs)
        a.square().mean().backward()
        torch.manual_seed(31)
        b = other.encode_scene(**kwargs)
        b.square().mean().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name)
                self.assertIsNotNone(q.grad,name)
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7)
        # Evaluation always takes the ordinary path, irrespective of the switch.
        model.eval(); other.eval()
        with torch.no_grad():
            torch.testing.assert_close(model.encode_scene(**kwargs),other.encode_scene(**kwargs),rtol=0,atol=0)

    def test_v27_only_changes_fusion_and_output_paths(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        a=OmegaConf.to_container(OmegaConf.load(root/'train_handrecon_v26_oracle_b_fullres.yaml'))
        b=OmegaConf.to_container(OmegaConf.load(root/'train_handrecon_v27_rgb_depth_tokens.yaml'))
        self.assertTrue(a['trainer']['model']['use_pointpainting'])
        self.assertFalse(b['trainer']['model']['use_pointpainting'])
        self.assertEqual(b['trainer']['model']['n_patches'],256)
        self.assertTrue(b['trainer']['model'].pop('fusion_activation_checkpointing'))
        b['trainer']['model']['use_pointpainting']=True
        for key in ('output_dir','log_file'):
            self.assertNotEqual(a['trainer']['train'][key],b['trainer']['train'][key])
            b['trainer']['train'][key]=a['trainer']['train'][key]
        # Resume is a per-run choice, not part of the architecture comparison.
        b['trainer']['train']['resume']=a['trainer']['train']['resume']
        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()

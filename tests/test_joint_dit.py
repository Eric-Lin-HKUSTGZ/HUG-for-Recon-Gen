import copy
import unittest
from pathlib import Path
import torch
from torch import nn
from omegaconf import OmegaConf
from src.models.joint_attention import AdaLNJointAttnBlock
from src.models.grasp_flow import TokenPerGroupDiT, GraspFlowMatching
from src.models.grasp_model import GraspFlowModel


def stats():
    return {k:{'mean':[0.]*n,'std':[1.]*n} for k,n in
            [('translation',3),('wrist_rot',6),('finger_rot',90),('shape',10)]}


def activate(model):
    # Nonzero gates/heads represent a trained model and expose both directions.
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'modulation' in name or name.startswith('out_'):
                p.normal_(0,0.08)


class JointDiTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self,joint_layers=2,checkpoint=False,dropout=0.0):
        return TokenPerGroupDiT(d_cond=24,d_model=32,n_layers=6,n_heads=4,
                               dropout=dropout,norm_stats=stats(),d_mano=109,
                               joint_layers=joint_layers,activation_checkpointing=checkpoint)

    def test_joint_block_identity_initialization_and_bidirectional_updates(self):
        block=AdaLNJointAttnBlock(32,4,0).eval()
        hand=torch.randn(2,4,32);context=torch.randn(2,8,32);time=torch.randn(2,32)
        h,c=block(hand,time,context)
        torch.testing.assert_close(h,hand,rtol=0,atol=0)
        torch.testing.assert_close(c,context,rtol=0,atol=0)
        activate(block)
        h,c=block(hand,time,context)
        h2,_=block(hand,time,context+torch.randn_like(context))
        _,c2=block(hand+torch.randn_like(hand),time,context)
        self.assertGreater((h-h2).abs().max().item(),1e-6)
        self.assertGreater((c-c2).abs().max().item(),1e-6)
        (h.square().mean()+c.square().mean()).backward()
        for stream in (block.hand,block.condition):
            for grad in stream.qkv.weight.grad.chunk(3,dim=0):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(),0)

    def test_denoiser_shape_no_condition_mutation_or_persistent_state(self):
        model=self.model().eval();activate(model)
        x=torch.randn(2,109);t=torch.rand(2);c=torch.randn(2,8,24);saved=c.clone()
        a=model(x,t,c)
        self.assertEqual(tuple(a.shape),(2,109))
        model(x+2,t,c+1)
        torch.testing.assert_close(a,model(x,t,c),rtol=0,atol=0)
        torch.testing.assert_close(c,saved,rtol=0,atol=0)
        a.square().mean().backward()
        self.assertEqual([n for n,p in model.named_parameters() if p.requires_grad and p.grad is None],[])

    def test_checkpointing_matches_dropout_outputs_and_gradients(self):
        a=self.model(dropout=0.1).train();activate(a);b=copy.deepcopy(a);b.activation_checkpointing=True
        x=torch.randn(2,109);t=torch.rand(2);c=torch.randn(2,8,24)
        torch.manual_seed(18);y=a(x,t,c);y.square().mean().backward()
        torch.manual_seed(18);z=b(x,t,c);z.square().mean().backward()
        torch.testing.assert_close(y,z,rtol=0,atol=0)
        for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertIsNotNone(q.grad,n)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7)

    def test_pretrained_tail_indices_retained_new_blocks_not_overwritten(self):
        old=self.model(joint_layers=0);new=self.model()
        wrapped=GraspFlowModel.__new__(GraspFlowModel);nn.Module.__init__(wrapped)
        wrapped.flow=nn.Module();wrapped.flow.denoise_fn=new
        before={k:v.clone() for k,v in new.joint_blocks.state_dict().items()}
        source={'flow.denoise_fn.'+k:v for k,v in old.state_dict().items()}
        incompatible,skipped=wrapped.load_compatible_state_dict(source)
        self.assertTrue(any(k.startswith('flow.denoise_fn.blocks.0.') for k,_ in skipped))
        self.assertTrue(any(k.startswith('flow.denoise_fn.joint_blocks.') for k in incompatible.missing_keys))
        for i in range(2,6):
            for k,v in new.blocks[i].state_dict().items():
                torch.testing.assert_close(v,old.blocks[i].state_dict()[k],rtol=0,atol=0)
        for k,v in new.joint_blocks.state_dict().items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertFalse(any(k.startswith('joint_blocks.') for k in old.state_dict()))
        with self.assertRaises(ValueError):self.model(joint_layers=7)

    def test_flow_ode_and_recovery_remain_109d(self):
        model=GraspFlowMatching(d_mano=109,d_cond=24,d_model=32,n_layers=6,n_heads=4,
                               norm_stats=stats(),joint_layers=2,dropout=0,sampling_steps=3)
        cond=torch.randn(2,8,24)
        result=model(torch.randn(2,109),cond)
        self.assertEqual(tuple(model.recover_x0(result).shape),(2,109))
        sample=model.eval().sample(cond)
        self.assertEqual(tuple(sample.shape),(2,109));self.assertTrue(torch.isfinite(sample).all())

    def test_config_changes_only_dit_and_outputs(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        a=OmegaConf.to_container(OmegaConf.load(root/'train_handrecon_v27_rgb_depth_tokens.yaml'))
        b=OmegaConf.to_container(OmegaConf.load(root/'train_handrecon_v28_joint_dit.yaml'))
        self.assertEqual(b['trainer']['model'].pop('flow_joint_layers'),2)
        self.assertTrue(b['trainer']['model'].pop('flow_activation_checkpointing'))
        for k in ('output_dir','log_file'):b['trainer']['train'][k]=a['trainer']['train'][k]
        self.assertIsNone(b['trainer']['train']['resume'])
        a['trainer']['train']['resume']=None  # v27 may be configured for an active resume
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main()

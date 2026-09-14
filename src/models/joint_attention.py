"""Single-frame Flux-style joint attention for hand and scene conditions.

Each stream has its own QKV, timestep modulation and MLP. Attention is over
[hand, scene] tokens jointly; both streams are updated inside a velocity call.
This module neither requires skeleton tokens nor carries state across ODE calls.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import GeLUMLP


class _JointStream(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = GeLUMLP(d_model, dropout)
        # scale1, shift1, gate1, scale2, shift2, gate2; identity at initialization.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def prepare(self, x, timestep):
        mods = self.modulation(timestep).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + mods[0]) + mods[1]
        qkv = self.qkv(h).reshape(x.shape[0], x.shape[1], 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return (self.q_norm(q), self.k_norm(k), v), mods

    def update(self, x, attended, mods):
        attended = attended.transpose(1, 2).reshape_as(x)
        x = x + mods[2] * self.out_proj(attended)
        h = self.norm2(x) * (1 + mods[3]) + mods[4]
        return x + mods[5] * self.mlp(h)


class AdaLNJointAttnBlock(nn.Module):
    """Separate hand/condition streams, one joint SDPA, two residual updates."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.hand = _JointStream(d_model, n_heads, dropout)
        self.condition = _JointStream(d_model, n_heads, dropout)
        self.dropout = dropout

    def forward(self, hand, c, context):
        hand_qkv, hand_mod = self.hand.prepare(hand, c)
        cond_qkv, cond_mod = self.condition.prepare(context, c)
        q, k, v = [torch.cat([h, s], dim=2) for h, s in zip(hand_qkv, cond_qkv)]
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        n_hand = hand.shape[1]
        return (
            self.hand.update(hand, attended[:, :, :n_hand], hand_mod),
            self.condition.update(context, attended[:, :, n_hand:], cond_mod),
        )

import copy
import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from src.models.encoders import DINOv2Encoder, LoRALinear
from src import train as train_module
from src.train import _build_optimizer, _strip_frozen_encoder


class _FakeSelfAttention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = torch.nn.Linear(width, width)
        self.key = torch.nn.Linear(width, width)
        self.value = torch.nn.Linear(width, width)

    def forward(self, values):
        return self.query(values) + self.key(values) + self.value(values)


class _FakeAttention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = _FakeSelfAttention(width)

    def forward(self, values):
        return self.attention(values)


class _FakeLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = _FakeAttention(width)
        self.gradient_checkpointing = False

    def forward(self, values):
        return values + self.attention(values)


class _FakeEncoder(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.layer = torch.nn.ModuleList([_FakeLayer(width) for _ in range(depth)])


class _FakeDINO(torch.nn.Module):
    def __init__(self, width=6, depth=12):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=width, patch_size=1, num_register_tokens=0
        )
        self.encoder = _FakeEncoder(width, depth)
        self.checkpoint_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs
        for layer in self.encoder.layer:
            layer.gradient_checkpointing = True

    def forward(self, values, return_dict=True):
        hidden = values
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        cls = torch.zeros(
            hidden.shape[0], 1, hidden.shape[-1],
            dtype=hidden.dtype, device=hidden.device,
        )
        return SimpleNamespace(last_hidden_state=torch.cat([cls, hidden], dim=1))


class _OptimizerModel(torch.nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.head = torch.nn.Linear(6, 2)


class DINOv2LoRATest(unittest.TestCase):
    def make_pair(self):
        torch.manual_seed(7)
        source = _FakeDINO()
        frozen_source = copy.deepcopy(source)
        lora_source = copy.deepcopy(source)
        tuning = {
            "mode": "lora",
            "layers": [8, 9, 10, 11],
            "target_modules": ["query", "value"],
            "rank": 2,
            "alpha": 4,
            "dropout": 0.05,
            "activation_checkpointing": True,
        }
        with patch(
            "src.models.encoders.AutoModel.from_pretrained",
            side_effect=[frozen_source, lora_source],
        ):
            frozen = DINOv2Encoder("fake", tuning={"mode": "frozen"})
            lora = DINOv2Encoder("fake", tuning=tuning)
        return frozen, lora

    def test_zero_initialized_lora_matches_frozen_and_backpropagates(self):
        frozen, lora = self.make_pair()
        values = torch.randn(2, 5, 6)
        frozen.eval()
        lora.eval()
        torch.testing.assert_close(frozen(values), lora(values), rtol=0, atol=0)

        lora.train()
        output = lora(values)
        output.square().mean().backward()
        lora_named = dict(lora.lora_named_parameters())
        self.assertEqual(len(lora_named), 16)
        self.assertTrue(
            all(parameter.requires_grad for parameter in lora_named.values())
        )
        self.assertTrue(
            any(
                name.endswith("lora_B")
                and parameter.grad is not None
                and parameter.grad.abs().sum().item() > 0
                for name, parameter in lora_named.items()
            )
        )
        frozen_base = [
            parameter
            for name, parameter in lora.named_parameters()
            if ".lora_A" not in name and ".lora_B" not in name
        ]
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen_base))
        self.assertTrue(all(parameter.grad is None for parameter in frozen_base))

    def test_injects_only_selected_qv_and_checkpoints_trainable_suffix(self):
        _, lora = self.make_pair()
        for index, layer in enumerate(lora.model.encoder.layer):
            attention = layer.attention.attention
            expected = index >= 8
            self.assertEqual(isinstance(attention.query, LoRALinear), expected)
            self.assertEqual(isinstance(attention.value, LoRALinear), expected)
            self.assertFalse(isinstance(attention.key, LoRALinear))
            self.assertEqual(layer.gradient_checkpointing, expected)
        self.assertEqual(lora.model.checkpoint_kwargs, {"use_reentrant": False})

    def test_optimizer_uses_separate_lora_learning_rate(self):
        _, lora = self.make_pair()
        model = _OptimizerModel(lora)
        config = OmegaConf.create(
            {
                "lr": 1e-4,
                "image_encoder_lr": 5e-5,
                "image_encoder_weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
            }
        )
        optimizer = _build_optimizer(model, config)
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(groups), {"main", "image_encoder_lora"})
        self.assertEqual(groups["main"]["lr_scale"], 1.0)
        self.assertEqual(groups["image_encoder_lora"]["lr_scale"], 0.5)
        self.assertEqual(groups["image_encoder_lora"]["weight_decay"], 0.0)

    def test_checkpoint_keeps_adapters_for_raw_and_ema_prefixes(self):
        state = {
            "head.weight": torch.ones(1),
            "image_encoder.model.encoder.layer.8.query.base.weight": torch.ones(1),
            "image_encoder.model.encoder.layer.8.query.lora_A": torch.ones(1),
            "module.image_encoder.model.encoder.layer.8.query.base.weight": torch.ones(1),
            "module.image_encoder.model.encoder.layer.8.query.lora_B": torch.ones(1),
            "n_averaged": torch.tensor(1),
        }
        kept = _strip_frozen_encoder(state)
        self.assertEqual(
            set(kept),
            {
                "head.weight",
                "image_encoder.model.encoder.layer.8.query.lora_A",
                "module.image_encoder.model.encoder.layer.8.query.lora_B",
                "n_averaged",
            },
        )

    def test_training_loop_clears_gradients_before_forward(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(train_module.main)))
        step_loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "step"
        ]
        self.assertEqual(len(step_loops), 1)

        step_loop = step_loops[0]
        zero_grad_lines = []
        forward_lines = []
        for node in ast.walk(step_loop):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "optimizer"
                and node.func.attr == "zero_grad"
            ):
                zero_grad_lines.append(node.lineno)
                self.assertTrue(
                    any(
                        keyword.arg == "set_to_none"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    )
                )
            if isinstance(node.func, ast.Name) and node.func.id == "ddp_model":
                forward_lines.append(node.lineno)

        self.assertEqual(len(zero_grad_lines), 1)
        self.assertEqual(len(forward_lines), 1)
        self.assertLess(zero_grad_lines[0], forward_lines[0])


if __name__ == "__main__":
    unittest.main()

"""Vision encoders: configurable DINOv2 and trainable PointNeXt."""

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .pointnext import PointNeXt


LOCAL_DINOV2_MODEL = "/root/code/vepfs/HUG-for-Recon-Gen/dinov2/hf_with_registers_base"


class LoRALinear(nn.Module):
    """Frozen linear projection plus a low-rank trainable residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            base.weight.new_empty((self.rank, base.in_features))
        )
        self.lora_B = nn.Parameter(
            base.weight.new_zeros((base.out_features, self.rank))
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder with an optional Q/V LoRA adapter."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-base",
        tuning: Optional[Dict] = None,
    ):
        super().__init__()
        # Saved legacy configs still use this Hub ID. Resolve it to the same
        # shared local weights so fresh cloud workers never require HF access.
        if model_name == "facebook/dinov2-with-registers-base":
            model_name = LOCAL_DINOV2_MODEL
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.tuning = dict(tuning or {})
        self.tuning_mode = str(self.tuning.get("mode", "frozen")).strip().lower()
        if self.tuning_mode not in {"frozen", "lora"}:
            raise ValueError(
                "image_encoder_tuning.mode must be 'frozen' or 'lora', "
                f"got {self.tuning_mode!r}"
            )
        self.is_trainable = self.tuning_mode == "lora"
        self.lora_layers = tuple()
        self.lora_target_modules = tuple()
        if self.is_trainable:
            self._inject_lora()
        else:
            self.model.eval()

    def _inject_lora(self) -> None:
        layers = self.model.encoder.layer
        selected = tuple(int(index) for index in self.tuning.get("layers", []))
        if not selected:
            raise ValueError("LoRA mode requires at least one layer index")
        if len(set(selected)) != len(selected):
            raise ValueError(f"LoRA layer indices must be unique, got {selected}")
        invalid = [index for index in selected if index < 0 or index >= len(layers)]
        if invalid:
            raise ValueError(
                f"LoRA layer indices {invalid} outside DINOv2 range 0..{len(layers) - 1}"
            )

        targets = tuple(
            str(name).strip().lower()
            for name in self.tuning.get("target_modules", ("query", "value"))
        )
        unsupported = sorted(set(targets).difference({"query", "key", "value"}))
        if unsupported or not targets:
            raise ValueError(
                "LoRA target_modules must be a non-empty subset of "
                f"query/key/value, got {targets}"
            )
        rank = int(self.tuning.get("rank", 8))
        alpha = float(self.tuning.get("alpha", 16.0))
        dropout = float(self.tuning.get("dropout", 0.0))
        for index in selected:
            attention = layers[index].attention.attention
            for name in targets:
                module = getattr(attention, name)
                if isinstance(module, LoRALinear):
                    raise ValueError(f"DINOv2 layer {index} {name} already has LoRA")
                setattr(
                    attention,
                    name,
                    LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
                )

        if bool(self.tuning.get("activation_checkpointing", True)):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Frozen prefix layers do not need recomputation. Restrict
            # checkpointing to the suffix that participates in LoRA backprop.
            first_trainable = min(selected)
            for index, layer in enumerate(layers):
                layer.gradient_checkpointing = index >= first_trainable

        self.lora_layers = selected
        self.lora_target_modules = targets

    def lora_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                yield name, parameter

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.is_trainable:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from an RGB image.

        Args:
            x: (B, 3, H, W) RGB images.

        Returns:
            (B, N, D) patch tokens, where N = (H/P) * (W/P) and D = hidden_size.
        """
        if self.is_trainable:
            outputs = self.model(x, return_dict=True)
        else:
            with torch.no_grad():
                outputs = self.model(x, return_dict=True)
        skip = 1 + self.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, skip:, :]
        return patch_tokens

    @property
    def output_dim(self) -> int:
        return self.hidden_size


class PointNeXtEncoder(nn.Module):
    """Trainable PointNeXt encoder.

    Wraps the PointNeXt U-Net to expose a consistent interface with
    ``DINOv2Encoder``. Outputs feature tokens AND metric XYZ centroids;
    fusion uses the centroids for the 3D Fourier positional embed.
    """

    def __init__(
        self,
        width: int = 64,
        sa_radii: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20),
        blocks: tuple[int, ...] = (1, 2, 1, 1),
        use_rgb: bool = True,
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.model = PointNeXt(
            c=width, sa_radii=sa_radii, blocks=blocks, use_rgb=use_rgb
        )

    def forward(self, xyz: torch.Tensor, rgb_pcl: Optional[torch.Tensor] = None):
        """Encode a point cloud into feature tokens and centroids.

        Args:
            xyz: (B, N, 3) metric meters.
            rgb_pcl: (B, N, 3) per-point RGB in [0, 1]. Required iff use_rgb.

        Returns:
            features: (B, 256, output_dim).
            centroids: (B, 256, 3).
        """
        return self.model(xyz, rgb_pcl=rgb_pcl)

    @property
    def output_dim(self) -> int:
        return self.model.out_dim

"""Full grasp model combining encoder, fusion, and flow matching."""

import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console

from ..utils.camera_geometry import backproject_pixels_torch
from ..utils.data_keys import MANO_RIGHT_SHAPE_FILE, NORM_STATS_FILE
from .encoders import DINOv2Encoder, PointNeXtEncoder
from .fusion import PatchFusion
from .grasp_flow import GraspFlowMatching
from .mano import MANO

console = Console()


class GraspFlowModel(nn.Module):
    """Full grasp model: encoder + fusion + flow matching."""

    def __init__(self, cfg, norm_stats=None):
        super().__init__()
        model_cfg = cfg.trainer.model
        self.d_mano = int(model_cfg.get("d_mano", 99))
        if self.d_mano not in (99, 109):
            raise ValueError(
                f"d_mano must be 99 (legacy) or 109 (learnable MANO shape), got {self.d_mano}"
            )
        self.use_rgb = model_cfg.get("use_rgb", True)
        self.use_depth = model_cfg.get("use_depth", True)
        self.use_2d_point = model_cfg.get("use_2d_point", False)
        use_pointpainting = model_cfg.get("use_pointpainting", True)
        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of use_rgb or use_depth must be true")

        # Coerce dependent flags: pointpainting needs both RGB and depth, and
        # 3D point conditioning needs depth-at-pixel + K, so any depth-off run
        # must use 2D point.
        if use_pointpainting and not (self.use_rgb and self.use_depth):
            console.print(
                "[yellow]use_pointpainting requires use_rgb and use_depth "
                "→ forcing use_pointpainting=false[/yellow]"
            )
            use_pointpainting = False
        if not self.use_depth and not self.use_2d_point:
            console.print(
                "[yellow]use_depth=false → forcing use_2d_point=true[/yellow]"
            )
            self.use_2d_point = True

        if norm_stats is None:
            if not NORM_STATS_FILE.exists():
                raise FileNotFoundError(
                    f"{NORM_STATS_FILE} not found. "
                    "Run: python -m src.utils.compute_norm_stats"
                )
            with open(NORM_STATS_FILE) as f:
                norm_stats = json.load(f)

        d_rgb_patch = 0
        d_depth_patch = 0
        if self.use_rgb:
            self.image_encoder = DINOv2Encoder(model_name=model_cfg.encoder_name)
            d_rgb_patch = self.image_encoder.output_dim
        if self.use_depth:
            sa_radii = tuple(model_cfg.get("pcl_sa_radii", (0.025, 0.05, 0.10, 0.20)))
            blocks = tuple(model_cfg.get("pcl_blocks", (1, 2, 1, 1)))
            self.pcl_use_rgb = model_cfg.get("pcl_use_rgb", True)
            self.depth_encoder = PointNeXtEncoder(
                width=model_cfg.get("pcl_width", 64),
                sa_radii=sa_radii,
                blocks=blocks,
                use_rgb=self.pcl_use_rgb,
            )
            d_depth_patch = self.depth_encoder.output_dim
        else:
            self.pcl_use_rgb = False

        d_fusion = model_cfg.d_fusion
        self.fusion = PatchFusion(
            d_rgb_patch=d_rgb_patch,
            d_depth_patch=d_depth_patch,
            d_model=d_fusion,
            n_patches=model_cfg.n_patches,
            n_layers=model_cfg.fusion_layers,
            n_heads=model_cfg.fusion_heads,
            dropout=model_cfg.dropout,
            patch_grid_size=model_cfg.patch_grid_size,
            use_rgb=self.use_rgb,
            use_depth=self.use_depth,
            use_pointpainting=use_pointpainting,
            image_size=model_cfg.get("image_size", 224),
            fourier_scale=model_cfg.get("fourier_scale", 1.0),
            use_2d_point=self.use_2d_point,
            # Missing in old configs by design: they retain exact legacy
            # query-fusion semantics when evaluating existing checkpoints.
            query_fusion_mode=model_cfg.get(
                "query_fusion_mode", "legacy_broadcast"
            ),
        )

        d_cond = d_fusion

        # Fixed MANO shape prior loaded from disk
        # Value: [-2.37, -1.25, -2.05, -0.85, 1.66, -1.35, -1.85, -0.67, -1.69, -1.21]
        fixed_betas = torch.from_numpy(np.load(MANO_RIGHT_SHAPE_FILE)).float()
        self.register_buffer("fixed_betas", fixed_betas.unsqueeze(0))

        self.mano = MANO()
        self.mesh_faces = self.mano.mano_layer.get_mano_closed_faces().cpu().numpy()

        self.flow = GraspFlowMatching(
            d_mano=self.d_mano,
            d_cond=d_cond,
            d_model=model_cfg.d_model,
            n_layers=model_cfg.flow_layers,
            n_heads=model_cfg.flow_heads,
            dropout=model_cfg.dropout,
            norm_stats=norm_stats,
            sampling_steps=model_cfg.get("sampling_steps", 50),
        )

    @staticmethod
    def _backproject(point_uv: torch.Tensor, camera_K: torch.Tensor) -> torch.Tensor:
        """Backproject (u, v, d) → metric (x, y, z) using K. Pure geometric op."""
        return backproject_pixels_torch(point_uv, camera_K)

    def encode_scene(
        self,
        point_uv: torch.Tensor,
        camera_K: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        pcl_xyz: Optional[torch.Tensor] = None,
        pcl_rgb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode scene inputs + query point into condition.

        Args:
            point_uv: (B, 3) = (u, v, depth_meters) at the query pixel. When
                use_2d_point, depth dim is ignored.
            camera_K: (B, 3, 3) intrinsics at the same resolution as point_uv.

        In 3D mode, backprojects (u, v, d) → metric xyz internally — K is used
        only for this geometric op, never as a learned input, so the model
        generalizes across cameras. In 2D mode, normalizes (u, v) by image_size
        and skips backprojection (no K or depth needed for point conditioning).

        Returns:
            cond: (B, d_fusion) pooled or (B, N, d_fusion) sequence.
        """
        if self.use_2d_point:
            point_coords = point_uv[:, :2] / float(self.fusion.image_size)
        else:
            point_coords = self._backproject(point_uv, camera_K)

        rgb_patches = None
        if self.use_rgb:
            with torch.no_grad():
                rgb_patches = self.image_encoder(rgb)

        depth_patches = None
        depth_centroids = None
        if self.use_depth:
            depth_patches, depth_centroids = self.depth_encoder(
                pcl_xyz, rgb_pcl=pcl_rgb if self.pcl_use_rgb else None
            )

        cond = self.fusion(
            point_coords,
            rgb_patches=rgb_patches,
            depth_patches=depth_patches,
            depth_centroids=depth_centroids,
            camera_K=camera_K,
        )
        return cond

    def mano_forward(self, mano_params, betas=None):
        """Run MANO and return landmarks + rotations in camera frame.

        In 109D mode the trailing ten entries are used as per-sample MANO
        shape coefficients. In legacy 99D mode the historical fixed shape is
        retained when ``betas`` is omitted.
        """
        if betas is None:
            betas = self.get_betas(mano_params)
        out = self.mano(mano_params, betas=betas)
        t = out["t"].unsqueeze(1)
        out["landmarks_3d"] = out["landmarks_3d"] + t
        out["vertices"] = out["vertices"] + t
        return out

    def get_betas(self, mano_params: torch.Tensor) -> torch.Tensor:
        """Return per-sample shape coefficients, with a 99D fallback.

        The fallback keeps old checkpoints and old 99D configs operational,
        while a 109D prediction remains differentiable with respect to shape.
        """
        if mano_params.shape[-1] >= 109:
            return mano_params[..., 99:109]
        expand_shape = (*mano_params.shape[:-1], 10)
        return self.fixed_betas.expand(expand_shape)

    def _build_dicts(
        self,
        pred_mano_params,
        params_norm_pred,
        params_norm_target,
        gt_mano_params,
    ):
        """Build pred/target dicts from predicted and GT mano params."""
        pred_out = self.mano_forward(pred_mano_params)
        preds = {
            "params_norm": params_norm_pred,
            "mano_params": pred_mano_params,
            "t": pred_out["t"],
            "R_3x3": pred_out["R_3x3"],
            "landmarks_3d": pred_out["landmarks_3d"],
            "vertices": pred_out["vertices"],
        }

        gt_out = self.mano_forward(gt_mano_params)
        targets = {
            "params_norm": params_norm_target,
            "mano_params": gt_mano_params,
            "t": gt_out["t"],
            "R_3x3": gt_out["R_3x3"],
            "landmarks_3d": gt_out["landmarks_3d"],
            "vertices": gt_out["vertices"],
        }
        return preds, targets

    def load_compatible_state_dict(self, state_dict):
        """Load matching weights across the legacy 99D/new 109D models.

        The only shape mismatch between the two denoisers is the token-type
        table (3 vs 4 rows). Existing translation/wrist/finger rows are copied
        and the new shape row/heads remain at their constructor initialization.
        Unexpected keys (e.g. the new shape heads when loading into a legacy
        model) are ignored. Other mismatches are reported and skipped rather
        than silently reshaping tensors.
        """
        current = self.state_dict()
        compatible = {}
        skipped = []
        for raw_key, value in state_dict.items():
            if raw_key == "n_averaged":
                continue
            key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
            if key not in current:
                skipped.append((key, "unexpected"))
                continue
            target = current[key]
            value = value.to(device=target.device, dtype=target.dtype)
            if value.shape == target.shape:
                compatible[key] = value
                continue
            if (
                key.endswith("token_type_emb")
                and value.ndim == target.ndim == 2
                and value.shape[1] == target.shape[1]
            ):
                merged = target.detach().clone()
                rows = min(value.shape[0], target.shape[0])
                merged[:rows] = value[:rows]
                compatible[key] = merged
                skipped.append((key, f"adapted {tuple(value.shape)} -> {tuple(target.shape)}"))
                continue
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(target.shape)}"))

        incompatible = self.load_state_dict(compatible, strict=False)
        return incompatible, skipped

    def forward(
        self,
        point_uv: torch.Tensor,
        camera_K: torch.Tensor,
        gt_mano_params: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        pcl_xyz: Optional[torch.Tensor] = None,
        pcl_rgb: Optional[torch.Tensor] = None,
    ):
        """Training forward pass. Returns (preds, targets, time_weight)."""
        scene = self.encode_scene(
            point_uv, camera_K, rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb
        )
        output = self.flow(gt_mano_params, scene)
        pred_mano_params = self.flow.recover_x0(output)

        preds, targets = self._build_dicts(
            pred_mano_params,
            output["pred"],
            output["target"],
            gt_mano_params,
        )
        time_weight = 1.0 - output["t"]
        return preds, targets, time_weight

    def build_loss_dicts(self, samples, gt_mano_params):
        """Build (preds, targets) dicts from validation samples."""
        preds_norm = self.flow.denoise_fn.normalize(samples)
        targets_norm = self.flow.denoise_fn.normalize(gt_mano_params)

        return self._build_dicts(
            samples,
            preds_norm,
            targets_norm,
            gt_mano_params,
        )

    @torch.no_grad()
    def sample(
        self,
        point_uv: torch.Tensor,
        camera_K: torch.Tensor,
        steps: Optional[int] = None,
        rgb: Optional[torch.Tensor] = None,
        pcl_xyz: Optional[torch.Tensor] = None,
        pcl_rgb: Optional[torch.Tensor] = None,
    ):
        """Generate grasp samples."""
        scene = self.encode_scene(
            point_uv, camera_K, rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb
        )
        return self.flow.sample(scene, steps=steps)

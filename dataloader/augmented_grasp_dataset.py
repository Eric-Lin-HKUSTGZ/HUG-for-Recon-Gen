"""Stage-1 training augmentations for the HUG hand-reconstruction dataset.

This module deliberately subclasses GraspDataset instead of changing the
original loader. Augmentations are enabled only for split="train" and are
applied through the loader's existing decode hooks, so RGB, depth, the sampled
query and the point cloud stay in the same coordinate frame.
"""

from typing import Dict, Optional

import numpy as np
import torch

from .grasp_dataset import GraspDataset


class AugmentedGraspDataset(GraspDataset):
    """GraspDataset with geometry-safe first-stage image/depth augmentation.

    The supported operations are intentionally conservative:

    * per-channel RGB scale, brightness and contrast;
    * small metric depth noise;
    * random depth dropout.

    The parent class then rebuilds point_uv and the point cloud from the
    augmented depth/RGB, while validation and inference remain unchanged.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        indices=None,
        image_size: int = 224,
        use_rgb: bool = True,
        use_depth: bool = True,
        samples_filename: Optional[str] = None,
        n_points_input: int = 4096,
        pcl_crop_radius: Optional[float] = 0.3,
        augmentation: Optional[Dict] = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            indices=indices,
            image_size=image_size,
            use_rgb=use_rgb,
            use_depth=use_depth,
            samples_filename=samples_filename,
            n_points_input=n_points_input,
            pcl_crop_radius=pcl_crop_radius,
        )
        self.augmentation = dict(augmentation or {})
        self.augmentation_enabled = bool(self.augmentation.get("enabled", False))
        # Set only while an augmented __getitem__ is executing. This keeps
        # get_inference_data()/get_original_for_viz() deterministic even when
        # the same dataset object is reused by an application.
        self._augmentation_state = None

    @staticmethod
    def _probability(value) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _new_augmentation_state(self) -> Dict[str, object]:
        """Sample one set of augmentation parameters for the current item."""
        color_scale = max(float(self.augmentation.get("color_scale", 0.0)), 0.0)
        lower = max(0.0, 1.0 - color_scale)
        upper = 1.0 + color_scale
        channel_scale = np.random.uniform(lower, upper, size=3).astype(np.float32)

        brightness_delta = max(
            float(self.augmentation.get("brightness_delta", 0.0)), 0.0
        )
        brightness = 0.0
        if np.random.random() < self._probability(
            self.augmentation.get("brightness_prob", 1.0)
        ):
            brightness = float(np.random.uniform(-brightness_delta, brightness_delta))

        contrast_min = float(self.augmentation.get("contrast_min", 1.0))
        contrast_max = float(self.augmentation.get("contrast_max", 1.0))
        if contrast_min > contrast_max:
            contrast_min, contrast_max = contrast_max, contrast_min
        contrast = 1.0
        if np.random.random() < self._probability(
            self.augmentation.get("contrast_prob", 1.0)
        ):
            contrast = float(np.random.uniform(contrast_min, contrast_max))

        return {
            "channel_scale": channel_scale,
            "brightness": brightness,
            "contrast": contrast,
            "depth_noise_std": max(
                float(self.augmentation.get("depth_noise_std", 0.0)), 0.0
            ),
            "depth_dropout_prob": self._probability(
                self.augmentation.get("depth_dropout_prob", 0.0)
            ),
        }

    @staticmethod
    def _apply_rgb_augmentation(rgb_np: np.ndarray, state: Dict[str, object]) -> np.ndarray:
        """Apply the sampled photometric transform and return uint8 RGB."""
        rgb = np.asarray(rgb_np, dtype=np.float32) / 255.0
        rgb = rgb * np.asarray(state["channel_scale"], dtype=np.float32)[None, None, :]
        rgb = rgb + float(state["brightness"])
        rgb = (rgb - 0.5) * float(state["contrast"]) + 0.5
        return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        rgb_np = GraspDataset._decode_image(image_bytes)
        state = self._augmentation_state
        if state is None:
            return rgb_np
        return self._apply_rgb_augmentation(rgb_np, state)

    def _depth_meters(self, depth_bytes: bytes) -> torch.Tensor:
        depth_m = GraspDataset._depth_meters(self, depth_bytes)
        state = self._augmentation_state
        if state is None:
            return depth_m

        depth = depth_m.numpy().copy()
        valid = depth > 0.0
        if valid.any() and float(state["depth_noise_std"]) > 0.0:
            noise = np.random.normal(
                0.0, float(state["depth_noise_std"]), size=int(valid.sum())
            ).astype(np.float32)
            depth[valid] = np.maximum(depth[valid] + noise, 0.0)

        dropout_prob = float(state["depth_dropout_prob"])
        if valid.any() and dropout_prob > 0.0:
            drop = valid & (np.random.random(size=depth.shape) < dropout_prob)
            # Keep at least one valid pixel so a very small/custom sample does
            # not produce an entirely empty cropped point cloud.
            if drop.all():
                valid_indices = np.flatnonzero(valid)
                keep = int(valid_indices[np.random.randint(valid_indices.size)])
                drop.flat[keep] = False
            depth[drop] = 0.0

        depth = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 100.0)
        return torch.from_numpy(depth.astype(np.float32, copy=False))

    def __getitem__(self, idx: int):
        # Validation/test datasets can use this class for structural symmetry,
        # but must never receive stochastic training augmentation.
        if not self.augmentation_enabled or self.split != "train":
            return super().__getitem__(idx)

        previous_state = self._augmentation_state
        self._augmentation_state = self._new_augmentation_state()
        try:
            return super().__getitem__(idx)
        finally:
            self._augmentation_state = previous_state


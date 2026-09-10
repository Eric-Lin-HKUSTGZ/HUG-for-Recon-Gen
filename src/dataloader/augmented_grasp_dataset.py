"""Stage-1 and stage-2 training augmentations for HUG reconstruction.

This module deliberately subclasses GraspDataset instead of changing the
original loader. Augmentations are enabled only for split="train" and are
applied through the loader's existing decode hooks, so RGB, depth, the sampled
query and the point cloud stay in the same coordinate frame. Stage-2 affine
augmentation additionally updates the camera intrinsics and image-space labels.
"""

from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .grasp_dataset import GraspDataset


class AugmentedGraspDataset(GraspDataset):
    """GraspDataset with geometry-safe RGB, structured-depth and PCL augmentation.

    The supported operations are intentionally conservative:

    * per-channel RGB scale, brightness and contrast;
    * metric depth noise, frame bias, holes and sparse outliers;
    * point-cloud jitter, density reduction and local holes;
    * one consistent affine transform for RGB/depth/mask, K, query and labels.

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
        d_mano: int = 99,
        query_min_depth: float = 0.15,
        query_max_depth: float = 2.0,
        query_depth_cluster_width: float = 0.12,
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
            d_mano=d_mano,
            query_min_depth=query_min_depth,
            query_max_depth=query_max_depth,
            query_depth_cluster_width=query_depth_cluster_width,
        )
        self.augmentation = dict(augmentation or {})
        self.augmentation_enabled = bool(self.augmentation.get("enabled", False))
        self.affine_cfg = dict(self.augmentation.get("affine", {}) or {})
        self.affine_enabled = bool(self.affine_cfg.get("enabled", False))
        self.depth_cfg = dict(self.augmentation.get("depth", {}) or {})
        self.pointcloud_cfg = dict(self.augmentation.get("pointcloud", {}) or {})
        # Set only while an augmented __getitem__ is executing. This keeps
        # get_inference_data()/get_original_for_viz() deterministic even when
        # the same dataset object is reused by an application.
        self._augmentation_state = None

    @staticmethod
    def _probability(value) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _uniform(value, default: float = 0.0) -> float:
        """Sample a scalar or a two-value [low, high] config range."""
        if value is None:
            return float(default)
        if (
            not isinstance(value, (str, bytes))
            and hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
        ):
            if not value:
                return float(default)
            if len(value) == 1:
                return float(value[0])
            low, high = float(value[0]), float(value[1])
            if low > high:
                low, high = high, low
            return float(np.random.uniform(low, high))
        return float(value)

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

        affine_params = None
        if self.affine_enabled and np.random.random() < self._probability(
            self.affine_cfg.get("probability", self.affine_cfg.get("prob", 1.0))
        ):
            scale_factor = max(float(self.affine_cfg.get("scale_factor", 0.15)), 0.0)
            rotation_deg = abs(float(self.affine_cfg.get("rotation_deg", 15.0)))
            translation_frac = max(
                float(self.affine_cfg.get("translation_frac", 0.02)), 0.0
            )
            affine_params = {
                "scale": float(
                    np.random.uniform(max(1.0 - scale_factor, 0.5), 1.0 + scale_factor)
                ),
                "angle_deg": float(np.random.uniform(-rotation_deg, rotation_deg)),
                "tx_frac": float(np.random.uniform(-translation_frac, translation_frac)),
                "ty_frac": float(np.random.uniform(-translation_frac, translation_frac)),
            }

        depth_active = bool(self.depth_cfg.get("enabled", True)) and (
            np.random.random()
            < self._probability(self.depth_cfg.get("probability", 1.0))
        )
        legacy_depth_std = self.augmentation.get("depth_noise_std", 0.0)
        legacy_depth_dropout = self.augmentation.get("depth_dropout_prob", 0.0)
        depth_noise_std = max(
            self._uniform(self.depth_cfg.get("gaussian_std", legacy_depth_std)), 0.0
        ) if depth_active else 0.0
        depth_dropout_prob = self._probability(
            self._uniform(
                self.depth_cfg.get("pixel_dropout_prob", legacy_depth_dropout)
            )
        ) if depth_active else 0.0
        frame_bias = 0.0
        if depth_active and np.random.random() < self._probability(
            self.depth_cfg.get("frame_bias_probability", 0.0)
        ):
            max_bias = abs(float(self.depth_cfg.get("frame_bias_max", 0.0)))
            frame_bias = float(np.random.uniform(-max_bias, max_bias))
        region_dropout = depth_active and np.random.random() < self._probability(
            self.depth_cfg.get("region_dropout_probability", 0.0)
        )

        pointcloud_active = bool(self.pointcloud_cfg.get("enabled", False)) and (
            np.random.random()
            < self._probability(self.pointcloud_cfg.get("probability", 1.0))
        )

        return {
            "channel_scale": channel_scale,
            "brightness": brightness,
            "contrast": contrast,
            "depth_noise_std": depth_noise_std,
            "depth_dropout_prob": depth_dropout_prob,
            "depth_frame_bias": frame_bias,
            "depth_region_dropout": region_dropout,
            "depth_region_size": self.depth_cfg.get("region_size", [5, 25]),
            "depth_outlier_prob": (
                self._probability(
                    self._uniform(self.depth_cfg.get("outlier_probability", 0.0))
                )
                if depth_active
                else 0.0
            ),
            "depth_outlier_delta": (
                abs(float(self.depth_cfg.get("outlier_delta", 0.0)))
                if depth_active
                else 0.0
            ),
            "pointcloud_active": pointcloud_active,
            "pointcloud_jitter_std": (
                max(self._uniform(self.pointcloud_cfg.get("jitter_std", 0.0)), 0.0)
                if pointcloud_active
                else 0.0
            ),
            "pointcloud_dropout_ratio": (
                self._probability(
                    self._uniform(self.pointcloud_cfg.get("dropout_ratio", 0.0))
                )
                if pointcloud_active
                else 0.0
            ),
            "pointcloud_region_dropout": (
                pointcloud_active
                and np.random.random()
                < self._probability(
                    self.pointcloud_cfg.get("region_dropout_probability", 0.0)
                )
            ),
            "pointcloud_region_radius": (
                max(
                    self._uniform(
                        self.pointcloud_cfg.get("region_radius", [0.01, 0.04])
                    ),
                    0.0,
                )
                if pointcloud_active
                else 0.0
            ),
            "affine_params": affine_params,
        }

    @staticmethod
    def _affine_matrix(shape, params: Dict[str, float]) -> np.ndarray:
        """Build an output-pixel transform around the image center.

        ``cv2.warpAffine`` samples source pixels with this matrix. The same
        homogeneous transform is left-multiplied into the camera intrinsics,
        which keeps projection, query points and 2D labels geometrically
        consistent after augmentation.
        """
        height, width = int(shape[0]), int(shape[1])
        center = ((width - 1.0) * 0.5, (height - 1.0) * 0.5)
        matrix = cv2.getRotationMatrix2D(
            center,
            float(params["angle_deg"]),
            float(params["scale"]),
        ).astype(np.float32)
        matrix[0, 2] += float(params["tx_frac"]) * width
        matrix[1, 2] += float(params["ty_frac"]) * height
        return matrix

    @staticmethod
    def _transform_points(points, matrix: np.ndarray):
        """Apply a 2x3 output-pixel transform to an arbitrary (...,2) array."""
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape[-1] < 2:
            return points
        flat = arr.reshape(-1, arr.shape[-1]).copy()
        xy1 = np.concatenate(
            [flat[:, :2], np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1
        )
        flat[:, :2] = xy1 @ matrix.T
        return flat.reshape(arr.shape)

    @staticmethod
    def _encode_rgb(rgb_np: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise ValueError("failed to encode affine-augmented RGB image")
        return encoded.tobytes()

    @staticmethod
    def _encode_png(image_np: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", image_np)
        if not ok:
            raise ValueError("failed to encode affine-augmented PNG")
        return encoded.tobytes()

    def _apply_affine_to_grasp_data(self, grasp_data: Dict, params: Dict[str, float]) -> Dict:
        """Warp all image-space fields in one copied sample dictionary."""
        # Pickle samples are freshly loaded for each item. Shallow-copy the
        # top-level and the two nested dictionaries we mutate instead of
        # deepcopying potentially large label arrays on every training item.
        data = dict(grasp_data)
        if isinstance(data.get("camera"), dict):
            data["camera"] = dict(data["camera"])
        if isinstance(data.get("grasp"), dict):
            data["grasp"] = dict(data["grasp"])
        rgb_np = GraspDataset._decode_image(data["image"])
        matrix = self._affine_matrix(rgb_np.shape[:2], params)
        height, width = rgb_np.shape[:2]
        # Do not keep an affine sample whose conditioning pixel leaves the
        # output frame. The parent loader clips only the depth lookup index,
        # while point_uv itself must remain in the same pixel frame.
        if data.get("condition_point") is not None:
            query_aug = self._transform_points(data["condition_point"], matrix)
            if (
                not np.isfinite(query_aug).all()
                or query_aug[0] < 0
                or query_aug[0] > width - 1
                or query_aug[1] < 0
                or query_aug[1] > height - 1
            ):
                return grasp_data
        # Training samples without a stored query still have GT 2D joints.
        # Reject transforms that clip most of the hand: otherwise RGB/depth
        # and labels remain mathematically consistent but the supervision is
        # dominated by out-of-frame joints and black border pixels.
        grasp = data.get("grasp")
        if isinstance(grasp, dict) and grasp.get("landmarks_2d") is not None:
            landmarks_aug = self._transform_points(grasp["landmarks_2d"], matrix)
            min_visible = float(
                np.clip(self.affine_cfg.get("min_visible_landmarks", 0.95), 0.0, 1.0)
            )
            visible = (
                np.isfinite(landmarks_aug).all(axis=-1)
                & (landmarks_aug[..., 0] >= 0)
                & (landmarks_aug[..., 0] <= width - 1)
                & (landmarks_aug[..., 1] >= 0)
                & (landmarks_aug[..., 1] <= height - 1)
            )
            if visible.mean() < min_visible:
                return grasp_data
        data["image"] = self._encode_rgb(
            cv2.warpAffine(
                rgb_np,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        )

        if data.get("depth"):
            depth_np = GraspDataset._decode_depth_uint16(data["depth"])
            dh, dw = depth_np.shape[:2]
            data["depth"] = self._encode_png(
                cv2.warpAffine(
                    depth_np,
                    matrix,
                    (dw, dh),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            )

        if data.get("object_mask"):
            mask_np = GraspDataset._decode_mask(data["object_mask"])
            mh, mw = mask_np.shape[:2]
            data["object_mask"] = self._encode_png(
                cv2.warpAffine(
                    mask_np,
                    matrix,
                    (mw, mh),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            )

        homography = np.eye(3, dtype=np.float32)
        homography[:2] = matrix
        camera = data.get("camera")
        if isinstance(camera, dict) and camera.get("K") is not None:
            camera["K"] = homography @ np.asarray(camera["K"], dtype=np.float32)

        if data.get("condition_point") is not None:
            data["condition_point"] = self._transform_points(
                data["condition_point"], matrix
            ).tolist()

        grasp = data.get("grasp")
        if isinstance(grasp, dict) and grasp.get("landmarks_2d") is not None:
            grasp["landmarks_2d"] = self._transform_points(
                grasp["landmarks_2d"], matrix
            )
        return data

    def _load_grasp_data(self, grasp_path):
        grasp_data = super()._load_grasp_data(grasp_path)
        state = self._augmentation_state
        if state is None or state.get("affine_params") is None:
            return grasp_data
        return self._apply_affine_to_grasp_data(
            grasp_data, state["affine_params"]
        )

    def get_augmented_for_viz(self, idx: int, seed: Optional[int] = None) -> Dict[str, object]:
        """Return one deterministic augmented sample for alignment inspection.

        This helper is intentionally separate from ``__getitem__`` and is not
        used by training. It exposes the exact image-space artifacts needed by
        the visualizer: augmented RGB/depth/mask, transformed camera matrix,
        labels and query point. ``seed`` makes the sampled augmentation
        parameters repeatable for a given item while restoring NumPy's global
        RNG state afterwards.
        """
        if not self.augmentation_enabled or self.split != "train":
            raise RuntimeError(
                "get_augmented_for_viz requires augmentation enabled and split='train'"
            )

        previous_state = self._augmentation_state
        previous_rng = np.random.get_state()
        if seed is not None:
            np.random.seed(int(seed))
        try:
            state = self._new_augmentation_state()
            self._augmentation_state = state
            path = self.grasp_files[idx]
            raw_original = GraspDataset._load_grasp_data(self, path)
            raw_augmented = self._load_grasp_data(path)
            # Call the parent implementation directly so this helper does not
            # recursively allocate a second augmentation state. The active
            # state is still consumed by the parent's dynamic decode hooks.
            item = GraspDataset.__getitem__(self, idx)
            rgb_augmented = self._decode_image(raw_augmented["image"])
            depth_augmented = self._depth_meters(raw_augmented["depth"]).numpy()
            mask_bytes = raw_augmented.get("object_mask")
            mask_augmented = (
                self._decode_mask(mask_bytes)
                if mask_bytes
                else np.zeros(depth_augmented.shape, dtype=np.uint8)
            )
            return {
                "item": item,
                "raw_original": raw_original,
                "raw_augmented": raw_augmented,
                "rgb_augmented": rgb_augmented,
                "depth_augmented": depth_augmented,
                "mask_augmented": mask_augmented,
                "augmentation_state": state,
            }
        finally:
            self._augmentation_state = previous_state
            np.random.set_state(previous_rng)

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
        valid = (depth > 0.0) & (depth < 3.0)
        frame_bias = float(state["depth_frame_bias"])
        if valid.any() and frame_bias != 0.0:
            depth[valid] = np.maximum(depth[valid] + frame_bias, 0.0)

        if valid.any() and float(state["depth_noise_std"]) > 0.0:
            noise = np.random.normal(
                0.0, float(state["depth_noise_std"]), size=int(valid.sum())
            ).astype(np.float32)
            depth[valid] = np.maximum(depth[valid] + noise, 0.0)

        if valid.any() and bool(state["depth_region_dropout"]):
            ys, xs = np.where(valid)
            center_idx = int(np.random.randint(len(ys)))
            cy, cx = int(ys[center_idx]), int(xs[center_idx])
            size_cfg = state["depth_region_size"]
            is_sequence = (
                not isinstance(size_cfg, (str, bytes))
                and hasattr(size_cfg, "__len__")
                and hasattr(size_cfg, "__getitem__")
            )
            min_size = max(int(size_cfg[0] if is_sequence else size_cfg), 1)
            max_size = max(
                int(size_cfg[1] if is_sequence and len(size_cfg) > 1 else min_size),
                min_size,
            )
            radius_y = int(np.random.randint(min_size, max_size + 1))
            radius_x = int(np.random.randint(min_size, max_size + 1))
            yy, xx = np.ogrid[: depth.shape[0], : depth.shape[1]]
            ellipse = (
                ((yy - cy) / max(radius_y, 1)) ** 2
                + ((xx - cx) / max(radius_x, 1)) ** 2
                <= 1.0
            )
            depth[ellipse & valid] = 0.0

        valid = (depth > 0.0) & (depth < 3.0)
        outlier_prob = float(state["depth_outlier_prob"])
        outlier_delta = float(state["depth_outlier_delta"])
        if valid.any() and outlier_prob > 0.0 and outlier_delta > 0.0:
            outliers = valid & (np.random.random(size=depth.shape) < outlier_prob)
            if outliers.any():
                depth[outliers] += np.random.uniform(
                    -outlier_delta, outlier_delta, size=int(outliers.sum())
                ).astype(np.float32)
                depth[outliers] = np.maximum(depth[outliers], 0.0)

        dropout_prob = float(state["depth_dropout_prob"])
        if valid.any() and dropout_prob > 0.0:
            drop = valid & (np.random.random(size=depth.shape) < dropout_prob)
            # Keep at least one valid pixel so a very small/custom sample does
            # not produce an entirely empty cropped point cloud.
            if drop[valid].all():
                valid_indices = np.flatnonzero(valid)
                keep = int(valid_indices[np.random.randint(valid_indices.size)])
                drop.flat[keep] = False
            depth[drop] = 0.0

        depth = np.clip(
            np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            3.0,
        )
        return torch.from_numpy(depth.astype(np.float32, copy=False))

    def _build_pcl(
        self,
        depth_m: torch.Tensor,
        rgb_np: np.ndarray,
        K: np.ndarray,
        point_xyz: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz, pcl_rgb = super()._build_pcl(
            depth_m,
            rgb_np,
            K,
            point_xyz=point_xyz,
            rng=rng,
        )
        state = self._augmentation_state
        if state is None or not bool(state.get("pointcloud_active", False)):
            return xyz, pcl_rgb

        xyz_np = xyz.numpy().copy()
        rgb_np_out = pcl_rgb.numpy().copy()
        valid = np.isfinite(xyz_np).all(axis=1) & (xyz_np[:, 2] > 0.0)
        if not valid.any():
            return xyz, pcl_rgb

        jitter_std = float(state["pointcloud_jitter_std"])
        if jitter_std > 0.0:
            xyz_np[valid] += np.random.normal(
                0.0, jitter_std, size=(int(valid.sum()), 3)
            ).astype(np.float32)

        drop = np.zeros(len(xyz_np), dtype=bool)
        if bool(state["pointcloud_region_dropout"]):
            valid_indices = np.flatnonzero(valid)
            center = xyz_np[int(np.random.choice(valid_indices))]
            radius = float(state["pointcloud_region_radius"])
            drop |= valid & (np.linalg.norm(xyz_np - center, axis=1) <= radius)

        dropout_ratio = float(state["pointcloud_dropout_ratio"])
        if dropout_ratio > 0.0:
            drop |= valid & (np.random.random(len(xyz_np)) < dropout_ratio)

        keep = valid & ~drop
        if keep.any() and drop.any():
            keep_indices = np.flatnonzero(keep)
            drop_indices = np.flatnonzero(drop)
            replacements = np.random.choice(
                keep_indices, size=len(drop_indices), replace=True
            )
            xyz_np[drop_indices] = xyz_np[replacements]
            rgb_np_out[drop_indices] = rgb_np_out[replacements]

        return torch.from_numpy(xyz_np), torch.from_numpy(rgb_np_out)

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

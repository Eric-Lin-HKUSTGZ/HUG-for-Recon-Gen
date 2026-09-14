"""Point cloud utilities.

Backproject depth + intrinsics into (N, 3) XYZ in metric meters with RGB.
Random-sample to a fixed point count for downstream PointNeXt FPS.
"""

from typing import Optional, Union

import numpy as np
import torch

from .camera_geometry import backproject_depth_np, backproject_pixels_np


def backproject_to_pcl(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    max_depth: float = 3.0,
    center: Optional[np.ndarray] = None,
    crop_radius: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject depth image to metric XYZ + RGB.

    Args:
        depth_m: (H, W) float32 depth in meters (0 = invalid).
        rgb: (H, W, 3) uint8 RGB at same resolution.
        K: (3, 3) intrinsics at the same resolution as depth.
        max_depth: clip points farther than this (meters).
        center: optional (3,) crop center in camera frame (meters).
        crop_radius: optional sphere radius (meters); keeps points within
            crop_radius of center. Both center and crop_radius must be set
            for the crop to apply; otherwise this filter is bypassed.

    Returns:
        xyz: (M, 3) float32 valid metric points.
        rgb_valid: (M, 3) uint8 colors aligned with xyz.
    """
    z = depth_m.astype(np.float32)
    valid = (z > 0) & (z < max_depth)
    xyz = backproject_depth_np(z, K).reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    m = valid.flatten()
    if center is not None and crop_radius is not None:
        center = np.asarray(center, dtype=np.float32).reshape(3)
        dist_sq = ((xyz - center) ** 2).sum(axis=-1)
        m = m & (dist_sq < crop_radius * crop_radius)
    return xyz[m], rgb_flat[m]


def sample_fixed_n(
    xyz: np.ndarray,
    rgb: np.ndarray,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample to exactly n_points. Random pick if more, repeat if fewer.

    Args:
        xyz: (M, 3) input points.
        rgb: (M, 3) colors.
        n_points: target count.
        rng: optional numpy RNG (uses default if None).
    """
    M = xyz.shape[0]
    if M == 0:
        return np.zeros((n_points, 3), np.float32), np.zeros((n_points, 3), np.uint8)
    choice = np.random.choice if rng is None else rng.choice
    if M >= n_points:
        idx = choice(M, size=n_points, replace=False)
    else:
        idx = choice(M, size=n_points, replace=True)
    return xyz[idx], rgb[idx]


def pixel_to_xyz(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
) -> np.ndarray:
    """Backproject a single pixel (u, v) at depth d (meters) through K → (x, y, z)."""
    point_uvd = np.array([u, v, depth], dtype=np.float32)
    return backproject_pixels_np(point_uvd, K)


def depth_to_pcl_tensors(
    depth_m: Union[torch.Tensor, np.ndarray],
    rgb: Union[torch.Tensor, np.ndarray],
    K: Union[torch.Tensor, np.ndarray],
    n_points: int = 4096,
    max_depth: float = 3.0,
    center: Optional[np.ndarray] = None,
    crop_radius: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end: depth + rgb + K → fixed-size (xyz, rgb) tensors.

    Inputs may be numpy or torch (CPU). Returns float32 torch tensors:
        xyz: (n_points, 3) meters
        rgb_pcl: (n_points, 3) in [0, 1]

    Args:
        center: optional (3,) crop center in camera frame (meters).
        crop_radius: optional sphere radius (meters) for object-centric
            crop. With no active sphere, sample valid pixels before backprojection;
            with a sphere, filter metric points before sampling as before.
    """
    if isinstance(depth_m, torch.Tensor):
        depth_m = depth_m.cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    if center is not None and crop_radius is not None:
        # Metric sphere membership must be determined before sampling.
        # Keep this legacy fallback unchanged, including its strict boundary.
        xyz_np, rgb_np = backproject_to_pcl(
            depth_m, rgb, K, max_depth=max_depth,
            center=center, crop_radius=crop_radius,
        )
        xyz_np, rgb_np = sample_fixed_n(xyz_np, rgb_np, n_points, rng=rng)
    else:
        # The keypoint ROI is already encoded as zero depth outside the circle.
        # Keep the same row-major candidate order and RNG call as sample_fixed_n,
        # but backproject only sampled pixels instead of the entire depth image.
        z = np.asarray(depth_m, dtype=np.float32)
        valid_idx = np.flatnonzero((z > 0) & (z < max_depth))
        count = valid_idx.size
        if count == 0:
            xyz_np = np.zeros((n_points, 3), dtype=np.float32)
            rgb_np = np.zeros((n_points, 3), dtype=np.uint8)
        else:
            choice = np.random.choice if rng is None else rng.choice
            selected = valid_idx[choice(count, size=n_points, replace=count < n_points)]
            v, u = np.divmod(selected, z.shape[1])
            uvd = np.stack([u, v, z.reshape(-1)[selected]], axis=-1).astype(np.float32)
            xyz_np = backproject_pixels_np(uvd, K)
            rgb_np = rgb.reshape(-1, 3)[selected]
    xyz = torch.from_numpy(xyz_np).float()
    rgb_pcl = torch.from_numpy(rgb_np).float() / 255.0
    return xyz, rgb_pcl

"""Camera projection helpers shared by training, point clouds, and viz.

The affine augmentation can produce a valid camera matrix with non-zero
off-diagonal entries (for example, an in-plane rotation introduces skew).
These helpers intentionally operate on the full 3x3 matrix instead of
assuming the special zero-skew pinhole form.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


def project_points_torch(
    points_3d: torch.Tensor,
    camera_K: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project (..., 3) camera-frame points with a (..., 3, 3) K matrix."""
    projected = torch.matmul(points_3d, camera_K.transpose(-1, -2))
    z = projected[..., 2:3].clamp_min(eps)
    return projected[..., :2] / z


def backproject_pixels_torch(
    point_uvd: torch.Tensor,
    camera_K: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Backproject (..., 3)=(u, v, depth_z) using the full K matrix.

    The third input is camera-space Z, rather than an arbitrary ray length.
    The solved ray is normalized by its third component before multiplication.
    """
    uv = point_uvd[..., :2]
    depth = point_uvd[..., 2:3]
    ones = torch.ones_like(depth)
    pixel_h = torch.cat([uv, ones], dim=-1)

    # K normally has one batch dimension fewer than point_uvd. Add singleton
    # dimensions so batched points (B, N, 3) also work with K shaped (B, 3, 3).
    solve_K = camera_K
    extra_batch_dims = point_uvd.ndim - (camera_K.ndim - 1)
    for _ in range(max(extra_batch_dims, 0)):
        solve_K = solve_K.unsqueeze(-3)
    rhs = pixel_h.unsqueeze(-1)
    if hasattr(torch.linalg, "solve"):
        ray = torch.linalg.solve(solve_K, rhs).squeeze(-1)
    else:
        ray = torch.solve(rhs, solve_K)[0].squeeze(-1)
    ray_z = ray[..., 2:3]
    safe_ray_z = torch.where(
        ray_z.abs() >= eps,
        ray_z,
        torch.where(
            ray_z >= 0,
            torch.full_like(ray_z, eps),
            torch.full_like(ray_z, -eps),
        ),
    )
    return ray * (depth / safe_ray_z)


def project_points_np(
    points_3d: np.ndarray,
    camera_K: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project (..., 3) camera-frame points with a full 3x3 K matrix."""
    points = np.asarray(points_3d, dtype=np.float64)
    K = np.asarray(camera_K, dtype=np.float64)
    projected = points @ K.T
    z = projected[..., 2:3]
    z = np.where(np.abs(z) >= eps, z, np.where(z >= 0, eps, -eps))
    return (projected[..., :2] / z).astype(np.float32)


def backproject_pixels_np(
    point_uvd: np.ndarray,
    camera_K: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Backproject (..., 3)=(u, v, depth_z) with a full 3x3 K matrix."""
    points = np.asarray(point_uvd, dtype=np.float64)
    K = np.asarray(camera_K, dtype=np.float64)
    flat = points.reshape(-1, 3)
    pixel_h = np.concatenate(
        [flat[:, :2], np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1
    )
    rays = np.linalg.solve(K, pixel_h.T).T
    ray_z = rays[:, 2:3]
    safe_ray_z = np.where(
        np.abs(ray_z) >= eps, ray_z, np.where(ray_z >= 0, eps, -eps)
    )
    xyz = rays * (flat[:, 2:3] / safe_ray_z)
    return xyz.reshape(points.shape).astype(np.float32)


def backproject_depth_np(
    depth_m: Union[np.ndarray, torch.Tensor],
    camera_K: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """Backproject an HxW depth-Z image to an HxWx3 camera-frame array."""
    if isinstance(depth_m, torch.Tensor):
        depth_m = depth_m.detach().cpu().numpy()
    if isinstance(camera_K, torch.Tensor):
        camera_K = camera_K.detach().cpu().numpy()
    depth = np.asarray(depth_m, dtype=np.float32)
    height, width = depth.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    uvd = np.stack([u, v, depth], axis=-1)
    return backproject_pixels_np(uvd, np.asarray(camera_K))

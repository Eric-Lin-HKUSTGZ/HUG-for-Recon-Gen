"""Visualize the exact train-time augmentation and check label alignment.

This is deliberately separate from ``visualize_dataloader.py``.  It uses
``AugmentedGraspDataset.get_augmented_for_viz`` so the rendered RGB/depth/mask,
camera matrix, 2D labels, query and PCL all come from one augmented sample.

Example (remote training machine):
    python scripts/visualize_augmented_dataloader.py --dataset dexycb --n 12
    python scripts/visualize_augmented_dataloader.py --dataset ho3d --n 12
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import tyro
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataloader.augmented_grasp_dataset import AugmentedGraspDataset  # noqa: E402
from src.dataloader.grasp_dataset import GraspDataset  # noqa: E402

console = Console()
ROOT = Path("/root/code/vepfs/dataset/hand_recon_hug")
DATASETS = {
    "dexycb": (
        ROOT / "dexycb_v2_canonical_right",
        ROOT / "splits_v2/dexycb_train.clean.txt",
    ),
    "ho3d": (
        ROOT / "ho3d",
        ROOT / "splits_v2/ho3d_train.clean.txt",
    ),
}

CHAINS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]
CHAIN_COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200)]

DEFAULT_AUGMENTATION = {
    "enabled": True,
    "color_scale": 0.20,
    "brightness_delta": 0.10,
    "brightness_prob": 0.50,
    "contrast_min": 0.90,
    "contrast_max": 1.10,
    "contrast_prob": 0.50,
    "depth_noise_std": 0.0015,
    "depth_dropout_prob": 0.03,
    "affine": {
        "enabled": True,
        "probability": 1.0,
        "scale_factor": 0.30,
        "rotation_deg": 30.0,
        "translation_frac": 0.02,
    },
}


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    proj = points @ K.T
    return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1]).copy()
    ones = np.ones((flat.shape[0], 1), dtype=np.float32)
    flat[:, :2] = np.concatenate([flat[:, :2], ones], axis=1) @ matrix.T
    return flat.reshape(arr.shape)


def draw_skeleton(img: np.ndarray, uv: np.ndarray) -> None:
    for chain, color in zip(CHAINS, CHAIN_COLORS):
        pts = uv[list(chain)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv2.LINE_AA)
    for point in uv:
        cv2.circle(img, tuple(point.astype(int)), 3, (255, 255, 255), -1)


def draw_query(img: np.ndarray, uv: np.ndarray) -> None:
    cv2.drawMarker(
        img,
        (int(round(uv[0])), int(round(uv[1]))),
        (0, 0, 255),
        cv2.MARKER_TILTED_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )


def draw_contour(img: np.ndarray, mask: np.ndarray, color) -> None:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, color, 1)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    normalized = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        normalized[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def add_label(img: np.ndarray, text: str, y: int = 14, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def render_sample(ds: AugmentedGraspDataset, idx: int, seed: int) -> tuple[np.ndarray, dict]:
    sample = ds.get_augmented_for_viz(idx, seed=seed)
    item = sample["item"]
    raw_original = sample["raw_original"]
    raw_augmented = sample["raw_augmented"]

    rgb_original = cv2.cvtColor(GraspDataset._decode_image(raw_original["image"]), cv2.COLOR_RGB2BGR)
    rgb_augmented = cv2.cvtColor(sample["rgb_augmented"], cv2.COLOR_RGB2BGR)
    depth_augmented = sample["depth_augmented"]
    mask_original = (
        GraspDataset._decode_mask(raw_original["object_mask"])
        if raw_original.get("object_mask")
        else np.zeros(depth_augmented.shape, np.uint8)
    )
    mask_augmented = sample["mask_augmented"]
    K_original = np.asarray(raw_original["camera"]["K"], dtype=np.float32)
    K_augmented = item["camera_K"].numpy()
    joints_3d = item["landmarks_3d"].numpy()
    joints_original = project(joints_3d, K_original)
    joints_augmented_projected = project(joints_3d, K_augmented)
    joints_augmented_label = (
        item["landmarks_2d"].numpy()
        if "landmarks_2d" in item
        else joints_augmented_projected
    )

    params = sample["augmentation_state"].get("affine_params")
    # The loader may reject an affine transform when it would clip too much
    # of the hand. Detect that case from the actual augmented camera matrix;
    # do not use the sampled-but-rejected parameters for the original overlay.
    affine_applied = not np.allclose(
        np.asarray(raw_augmented["camera"]["K"], dtype=np.float32),
        K_original,
        atol=1e-5,
    )
    matrix = (
        ds._affine_matrix(rgb_original.shape[:2], params)
        if params is not None and affine_applied
        else np.eye(2, 3, dtype=np.float32)
    )
    query_augmented = item["point_uv"].numpy()[:2]
    inverse = cv2.invertAffineTransform(matrix)
    query_original = transform_points(query_augmented[None], inverse)[0]

    # Panel 1: unmodified image and labels. Panel 2: augmented image and the
    # labels after the same affine transform. The latter is the main alignment
    # check requested for training-time augmentation.
    p1 = rgb_original.copy()
    draw_contour(p1, mask_original, (0, 255, 0))
    draw_skeleton(p1, joints_original)
    draw_query(p1, query_original)
    add_label(p1, "original RGB + GT/mask/query")

    p2 = rgb_augmented.copy()
    draw_contour(p2, mask_augmented, (0, 255, 0))
    draw_skeleton(p2, joints_augmented_projected)
    for point in joints_augmented_label:
        cv2.circle(p2, tuple(point.astype(int)), 2, (255, 255, 0), -1)
    draw_query(p2, query_augmented)
    add_label(p2, "augmented RGB + projected GT / labels")

    sampling = ds._query_sampling_debug(mask_augmented, depth_augmented)
    p3 = colorize_depth(depth_augmented)
    draw_contour(p3, sampling["component"], (0, 255, 0))
    draw_contour(p3, sampling["core"], (255, 0, 0))
    draw_query(p3, query_augmented)
    add_label(p3, "augmented depth + mask/core/query")

    p4 = cv2.cvtColor((sampling["component"] > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    draw_contour(p4, sampling["core"], (255, 0, 0))
    draw_contour(p4, sampling["candidate"], (255, 0, 255))
    draw_skeleton(p4, joints_augmented_label)
    draw_query(p4, query_augmented)
    add_label(p4, "augmented mask + GT/core/candidates")

    p5 = rgb_augmented.copy()
    pcl = item.get("pcl_xyz")
    if pcl is not None:
        pcl = pcl.numpy()
        puv = project(pcl, K_augmented)
        z = pcl[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        colors = cv2.applyColorMap(
            (np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        ).reshape(-1, 3)
        in_frame = (puv[:, 0] >= 0) & (puv[:, 0] < 224) & (puv[:, 1] >= 0) & (puv[:, 1] < 224)
        for point, color in zip(puv[in_frame], colors[in_frame]):
            cv2.circle(p5, tuple(point.astype(int)), 1, tuple(int(x) for x in color), -1)
    draw_query(p5, query_augmented)
    add_label(p5, "augmented RGB + PCL projection")

    reproj_error = np.linalg.norm(joints_augmented_projected - joints_augmented_label, axis=1)
    expected_augmented_label = transform_points(joints_original, matrix)
    label_transform_error = np.linalg.norm(
        expected_augmented_label - joints_augmented_label, axis=1
    )
    qx = int(np.clip(round(query_augmented[0]), 0, 223))
    qy = int(np.clip(round(query_augmented[1]), 0, 223))
    joints_xy = np.rint(joints_augmented_label).astype(int)
    in_frame = (
        (joints_xy[:, 0] >= 0) & (joints_xy[:, 0] < 224)
        & (joints_xy[:, 1] >= 0) & (joints_xy[:, 1] < 224)
    )
    in_mask = np.zeros(len(joints_xy), dtype=bool)
    in_mask[in_frame] = mask_augmented[joints_xy[in_frame, 1], joints_xy[in_frame, 0]] > 0
    original_xy = np.rint(joints_original).astype(int)
    original_frame = (
        (original_xy[:, 0] >= 0) & (original_xy[:, 0] < 224)
        & (original_xy[:, 1] >= 0) & (original_xy[:, 1] < 224)
    )
    original_in_mask = np.zeros(len(original_xy), dtype=bool)
    original_in_mask[original_frame] = mask_original[
        original_xy[original_frame, 1], original_xy[original_frame, 0]
    ] > 0
    metrics = {
        "stem": item["stem"],
        "affine": params,
        "affine_applied": affine_applied,
        "reprojection_mean_px": float(reproj_error.mean()),
        "reprojection_max_px": float(reproj_error.max()),
        "label_transform_mean_px": float(label_transform_error.mean()),
        "label_transform_max_px": float(label_transform_error.max()),
        "joint_in_frame": float(in_frame.mean()),
        "joint_in_mask": float(in_mask.mean()),
        "original_joint_in_mask": float(original_in_mask.mean()),
        "query_in_mask": bool(mask_augmented[qy, qx] > 0),
        "query_in_core": bool(sampling["core"][qy, qx] > 0),
        "candidate_px": int(sampling["candidate"].sum()),
        "fallback_stage": sampling["fallback_stage"],
    }
    panel = np.concatenate([p1, p2, p3, p4, p5], axis=1)
    status = (
        f"K/reproj {metrics['reprojection_mean_px']:.3f}/{metrics['reprojection_max_px']:.3f}px | "
        f"affine-label {metrics['label_transform_mean_px']:.3f}px | "
        f"joint-mask {metrics['original_joint_in_mask'] * 100:.0f}%→{metrics['joint_in_mask'] * 100:.0f}% | "
        f"query mask/core {int(metrics['query_in_mask'])}/{int(metrics['query_in_core'])} | "
        f"affine {'on' if metrics['affine_applied'] else 'rejected'}"
    )
    cv2.putText(panel, status, (4, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    return panel, metrics


def main(
    dataset: str,
    n: int = 12,
    out: Path = Path("/root/code/vepfs/HUG-for-Recon-Gen/viz_augmented"),
    seed: int = 17,
    cols: int = 2,
    indices: str | None = None,
):
    """Render deterministic train-time augmentation checks for DexYCB or HO3D."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {tuple(DATASETS)}")
    data_path, split_file = DATASETS[dataset]
    ds = AugmentedGraspDataset(
        str(data_path),
        split="train",
        samples_filename=str(split_file),
        use_rgb=True,
        use_depth=True,
        n_points_input=4096,
        pcl_crop_radius=0.2,
        d_mano=109,
        augmentation=DEFAULT_AUGMENTATION,
    )
    if indices:
        idxs = [int(x.strip()) for x in indices.split(",") if x.strip()]
    else:
        idxs = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
        idxs = sorted(int(x) for x in idxs)
    if any(i < 0 or i >= len(ds) for i in idxs):
        raise IndexError(f"indices must be in [0, {len(ds)})")

    out.mkdir(parents=True, exist_ok=True)
    panels, metrics = [], []
    for rank, idx in enumerate(idxs):
        panel, metric = render_sample(ds, idx, seed + rank)
        panels.append(panel)
        metrics.append(metric)
        image_path = out / f"{dataset}_augmented_{rank:03d}.png"
        cv2.imwrite(str(image_path), panel)
        console.print(
            f"[cyan]{image_path.name}[/cyan] idx={idx} stem={metric['stem']} "
            f"reproj={metric['reprojection_mean_px']:.3f}/{metric['reprojection_max_px']:.3f}px "
            f"affine-label={metric['label_transform_mean_px']:.3f}px "
            f"joint-mask={metric['original_joint_in_mask'] * 100:.0f}%→{metric['joint_in_mask'] * 100:.0f}% "
            f"query-mask/core={int(metric['query_in_mask'])}/{int(metric['query_in_core'])} "
            f"affine={'on' if metric['affine_applied'] else 'rejected'}"
        )

    ph, pw = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    grid = np.full((rows * (ph + 8), cols * (pw + 8), 3), 24, np.uint8)
    for k, panel in enumerate(panels):
        row, col = divmod(k, cols)
        grid[row * (ph + 8): row * (ph + 8) + ph, col * (pw + 8): col * (pw + 8) + pw] = panel
    grid_path = out / f"{dataset}_augmented_grid.png"
    cv2.imwrite(str(grid_path), grid)
    console.print(f"[bold green]grid -> {grid_path}[/bold green]")
    mean_reproj = np.mean([m["reprojection_mean_px"] for m in metrics])
    mean_mask = np.mean([m["joint_in_mask"] for m in metrics])
    mean_transform = np.mean([m["label_transform_mean_px"] for m in metrics])
    query_ok = sum(bool(m["query_in_mask"] and m["query_in_core"]) for m in metrics)
    console.print(
        f"[bold]summary:[/bold] mean reprojection={mean_reproj:.4f}px, "
        f"mean affine-label error={mean_transform:.4f}px, "
        f"mean joint-in-mask={mean_mask * 100:.1f}%, "
        f"query in mask/core={query_ok}/{len(metrics)}"
    )


if __name__ == "__main__":
    tyro.cli(main)

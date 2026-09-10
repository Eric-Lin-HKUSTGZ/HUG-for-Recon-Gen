"""Visual sanity check for the dataloader (GraspDataset) on converted datasets.

Renders per-sample panels so you can eyeball that RGB / GT joints / mask /
query point / PCL are mutually aligned. Uses the REAL GraspDataset loading
path - what you see is what the model gets.

Panels per sample (left to right):
  1. RGB + GT skeleton (projected landmarks_3d via camera_K) + mask contour
     (green) + query point (red cross) + stored landmarks_2d (cyan dots)
  2. RGB + backprojected PCL points (colored by depth, see colorbar note)
  3. Depth map (viridis) + mask contour + query point
  4. Hand mask + skeleton + query point

Also prints quantitative sanity metrics per sample (out-of-frame joint
fraction, query-depth vs wrist-z gap, PCL/wrist distance stats).

Usage:
    python scripts/visualize_dataloader.py --dataset dexycb --split train --n 16
    python scripts/visualize_dataloader.py --dataset ho3d --split train --n 16
    python scripts/visualize_dataloader.py --dataset ho3d_eval --split eval --n 16
    python scripts/visualize_dataloader.py --dataset dexycb --split test --n 32 \
        --out /root/code/vepfs/HUG-for-Recon-Gen/viz --seed 7
    # 精确检查某个清单索引（可用日志中的 dataset index 复现）
    python scripts/visualize_dataloader.py --dataset dexycb --split train --indices 0,10,100 \
        --out /root/code/vepfs/HUG-for-Recon-Gen/viz/suspicious
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import tyro
from rich.console import Console

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataloader.grasp_dataset import GraspDataset  # noqa: E402

console = Console()

ROOT = Path("/root/code/vepfs/dataset/hand_recon_hug")
DATASETS = {
    "dexycb": (
        ROOT / "dexycb",
        {
            "train": ROOT / "splits_v2/dexycb_train.clean.txt",
            "val": ROOT / "splits_v2/dexycb_val.clean.txt",
            "test": ROOT / "splits_v2/dexycb_test.clean.txt",
        },
    ),
    "ho3d": (
        ROOT / "ho3d",
        {
            "train": ROOT / "splits/ho3d_train.clean.txt",
            "val": ROOT / "splits/ho3d_val.clean.txt",
        },
    ),
    "ho3d_eval": (
        ROOT / "ho3d_eval",
        {"eval": ROOT / "splits/ho3d_eval.clean.txt"},
    ),
}

# 21-joint skeleton in our standard order (manotorch):
# wrist, thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
CHAINS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]
CHAIN_COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200)]

IMAGENET_MEAN = np.array(GraspDataset.IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD = np.array(GraspDataset.IMAGENET_STD, dtype=np.float32)


def denormalize_rgb(t) -> np.ndarray:
    """ImageNet-normalized (3,224,224) tensor -> uint8 BGR image."""
    img = t.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def project(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,3) camera-frame -> (N,2) pixels."""
    proj = points_3d @ K.T
    return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)


def draw_skeleton(img: np.ndarray, uv: np.ndarray, scale: float = 1.0) -> None:
    for chain, color in zip(CHAINS, CHAIN_COLORS):
        pts = uv[list(chain)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(
                img, tuple(a.astype(int)), tuple(b.astype(int)), color,
                max(1, int(2 * scale)), cv2.LINE_AA,
            )
    for p in uv:
        cv2.circle(img, tuple(p.astype(int)), max(2, int(3 * scale)), (255, 255, 255), -1)


def draw_query(img: np.ndarray, u: float, v: float) -> None:
    p = (int(u), int(v))
    cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)


def project_crop_sphere_hull(
    center_xyz: np.ndarray, radius_m: float, K: np.ndarray, image_size: int
) -> np.ndarray | None:
    """Project the 3D crop sphere and return its 2D convex-hull contour.

    The PCL crop is a sphere in camera XYZ, not a 2D image circle. Sampling
    its surface and taking the projected convex hull gives a faithful overlay
    for the RGB/PCL panel (perspective projection can make the silhouette
    slightly non-circular). Points behind the camera are discarded.
    """
    samples = []
    n_lat, n_lon = 18, 36
    for lat in np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat):
        cos_lat, sin_lat = np.cos(lat), np.sin(lat)
        for lon in np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False):
            samples.append(
                center_xyz
                + radius_m
                * np.array(
                    [cos_lat * np.cos(lon), cos_lat * np.sin(lon), sin_lat],
                    dtype=np.float32,
                )
            )
    sphere = np.asarray(samples, dtype=np.float32)
    sphere = sphere[sphere[:, 2] > 1e-4]
    if len(sphere) < 3:
        return None
    uv = project(sphere, K)
    inb = (
        (uv[:, 0] >= -image_size)
        & (uv[:, 0] <= 2 * image_size)
        & (uv[:, 1] >= -image_size)
        & (uv[:, 1] <= 2 * image_size)
    )
    uv = uv[inb]
    if len(uv) < 3:
        return None
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    return hull.reshape(-1, 2)


def draw_crop_sphere(
    img: np.ndarray, center_xyz: np.ndarray, radius_m: float, K: np.ndarray
) -> None:
    """Draw the projected 3D crop-sphere boundary and its center."""
    hull = project_crop_sphere_hull(center_xyz, radius_m, K, img.shape[1])
    if hull is not None:
        cv2.polylines(img, [hull], isClosed=True, color=(0, 255, 255), thickness=2)
    uv_center = project(center_xyz.reshape(1, 3), K)[0]
    draw_query(img, float(uv_center[0]), float(uv_center[1]))
    cv2.putText(
        img,
        f"crop R={radius_m * 100:.0f}cm",
        (4, img.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_mask_contour(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> None:
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, cnts, -1, color, 1)


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    d = depth_m.copy()
    valid = d > 0
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def render_sample(ds: GraspDataset, idx: int) -> tuple[np.ndarray, dict]:
    item = ds[idx]
    K = item["camera_K"].numpy()
    rgb = denormalize_rgb(item["rgb"]) if "rgb" in item else np.zeros((224, 224, 3), np.uint8)
    stem = item["stem"]
    u, v, qd = (float(x) for x in item["point_uv"])

    # GT 3D joints: train pkls store landmarks_3d (canonical shape); eval pkls
    # carry joints_gt (already reordered by the dataset).
    if "landmarks_3d" in item:
        joints_3d = item["landmarks_3d"].numpy()
        gt_kind = "mano"
    else:
        joints_3d = item["joints_gt"].numpy()
        gt_kind = "joints_gt"
    j2d = project(joints_3d, K)

    # raw pkl for the mask (dataset does not return it)
    raw = ds._load_grasp_data(ds.grasp_files[idx])
    mask = ds._decode_mask(raw["object_mask"]) if raw.get("object_mask") else np.zeros((224, 224), np.uint8)

    p1 = rgb.copy()
    draw_mask_contour(p1, mask)
    draw_skeleton(p1, j2d)
    if "landmarks_2d" in item:  # stored 2D landmarks for cross-check
        for p in item["landmarks_2d"].numpy():
            cv2.circle(p1, tuple(p.astype(int)), 1, (255, 255, 0), -1)
    draw_query(p1, u, v)

    p2 = rgb.copy()
    q_xyz = np.array(
        [
            (u - K[0, 2]) * qd / K[0, 0],
            (v - K[1, 2]) * qd / K[1, 1],
            qd,
        ],
        dtype=np.float32,
    )
    crop_radius_m = float(ds.pcl_crop_radius) if ds.pcl_crop_radius is not None else None
    if "pcl_xyz" in item:
        pcl = item["pcl_xyz"].numpy()
        puv = project(pcl, K)
        z = pcl[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        zn = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        colors = cv2.applyColorMap((zn * 255).astype(np.uint8).reshape(-1, 1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
        inb = (puv[:, 0] >= 0) & (puv[:, 0] < 224) & (puv[:, 1] >= 0) & (puv[:, 1] < 224)
        for p, c in zip(puv[inb], colors[inb]):
            cv2.circle(p2, tuple(p.astype(int)), 1, tuple(int(x) for x in c), -1)
        # PCL points are already filtered by _build_pcl. The crop sphere is
        # visualized as a diagnostic region, but use the same camera-frame
        # center that _build_pcl receives.
        if crop_radius_m is not None:
            d_crop = np.linalg.norm(pcl - q_xyz, axis=1)
            crop_max_dist_cm = float(d_crop.max() * 100.0)
            crop_inside_frac = float((d_crop <= crop_radius_m + 1e-5).mean())
    # Yellow contour = the actual 3D sphere used by _build_pcl for the PCL crop;
    # red cross = query pixel. They coincide only when the query is depth-correct.
    if crop_radius_m is not None:
        draw_crop_sphere(p2, q_xyz, crop_radius_m, K)
    draw_query(p2, u, v)

    depth_m = ds._depth_meters(raw["depth"]).numpy()
    sampling = ds._query_sampling_debug(mask, depth_m)
    p3 = colorize_depth(depth_m)
    draw_mask_contour(p3, sampling["component"], color=(0, 255, 0))
    draw_mask_contour(p3, sampling["core"], color=(255, 0, 0))
    draw_query(p3, u, v)

    p4 = cv2.cvtColor((sampling["component"] > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    draw_mask_contour(p4, sampling["core"], color=(255, 0, 0))
    draw_mask_contour(p4, sampling["candidate"], color=(255, 0, 255))
    draw_skeleton(p4, j2d)
    draw_query(p4, u, v)
    cv2.putText(
        p4,
        "green=mask blue=core magenta=candidates",
        (4, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for panel, label in zip(
        (p1, p2, p3, p4),
        ("RGB+GT+mask+query", "RGB+PCL", "depth+mask+query", "mask+GT"),
    ):
        cv2.putText(panel, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.concatenate([p1, p2, p3, p4], axis=1)
    cv2.putText(
        panel, f"{stem} [{gt_kind}]", (4, 236),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
    )

    # ---- quantitative sanity metrics ----
    out_frac = float(
        ((j2d[:, 0] < 0) | (j2d[:, 0] >= 224) | (j2d[:, 1] < 0) | (j2d[:, 1] >= 224)).mean()
    )
    wrist_z = float(joints_3d[0, 2])
    qx, qy = int(np.clip(round(u), 0, 223)), int(np.clip(round(v), 0, 223))
    m = {
        "stem": stem,
        "out_frac": out_frac,
        "mask_px": int((mask > 0).sum()),
        "core_px": int((sampling["core"] > 0).sum()),
        "candidate_px": int((sampling["candidate"] > 0).sum()),
        "query_in_mask": bool(sampling["component"][qy, qx] > 0),
        "query_in_core": bool(sampling["core"][qy, qx] > 0),
        "query_depth_consistent": bool(sampling["depth_consistent"][qy, qx] > 0),
        "fallback_stage": sampling["fallback_stage"],
        "erode_iters": int(sampling["erode_iters"]),
        "query_d": qd,
        "wrist_z": wrist_z,
        "q_minus_wrist_mm": (qd - wrist_z) * 1000.0,
    }
    if "pcl_xyz" in item:
        d_wrist = np.linalg.norm(item["pcl_xyz"].numpy() - joints_3d[0], axis=1)
        m["pcl_near_wrist_frac"] = float((d_wrist < 0.15).mean())
        if crop_radius_m is not None:
            m["pcl_crop_max_dist_cm"] = crop_max_dist_cm
            m["pcl_crop_inside_frac"] = crop_inside_frac
    return panel, m


def main(
    dataset: str,
    split: str = "train",
    n: int = 16,
    out: Path = Path("/root/code/vepfs/HUG-for-Recon-Gen/viz"),
    seed: int = 0,
    cols: int = 2,
    indices: str | None = None,
    crop_radius: float = 0.2,
    data_path: Path | None = None,
    split_dir: Path | None = None,
):
    """Render sanity panels for random dataloader samples.

    Args:
        dataset: dexycb | ho3d | ho3d_eval
        split: train | val | test | eval (mapped to splits/*.clean.txt)
        n: number of samples to render
        out: output directory for PNG panels + grid
        seed: sampling seed
        cols: panels per row in the summary grid image
        indices: comma-separated exact dataset indices; overrides random sampling
        crop_radius: 3D PCL crop radius in meters; should match the training config
        data_path: override the converted pkl directory for the selected dataset
        split_dir: override the directory containing train/val/test stem lists
    """
    default_data_path, lists = DATASETS[dataset]
    if data_path is None:
        data_path = default_data_path
    data_path = Path(data_path)
    if split_dir is not None:
        split_dir = Path(split_dir)
        lists = {name: split_dir / Path(path).name for name, path in lists.items()}
    ds = GraspDataset(
        str(data_path),
        samples_filename=str(lists[split]),
        split="val",
        use_rgb=True,
        use_depth=True,
        n_points_input=4096,
        pcl_crop_radius=crop_radius,
    )
    rng = np.random.default_rng(seed)
    if indices:
        idxs = [int(x.strip()) for x in indices.split(",") if x.strip()]
        if any(i < 0 or i >= len(ds) for i in idxs):
            raise IndexError(f"indices must be in [0, {len(ds)})")
    else:
        idxs = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist())

    out.mkdir(parents=True, exist_ok=True)
    panels, metrics = [], []
    for rank, i in enumerate(idxs):
        panel, m = render_sample(ds, i)
        panels.append(panel)
        metrics.append(m)
        fp = out / f"{dataset}_{split}_{rank:03d}.png"
        cv2.imwrite(str(fp), panel)
        console.print(
            f"[cyan]{fp.name}[/cyan]  idx={i}  stem={m['stem']}  "
            f"关节出画 {m['out_frac'] * 100:.0f}%  mask {m['mask_px']}px  "
            f"query深度 {m['query_d']:.3f}m  腕z {m['wrist_z']:.3f}m  "
            f"差 {m['q_minus_wrist_mm']:+.0f}mm  "
            f"core/cand {m['core_px']}/{m['candidate_px']}px  "
            f"inside {int(m['query_in_mask'])}/{int(m['query_in_core'])}/{int(m['query_depth_consistent'])}  "
            f"fallback {m['fallback_stage']}  "
            f"PCL近腕(<15cm) {m.get('pcl_near_wrist_frac', 0.0) * 100:.0f}%  "
            f"crop内 {m.get('pcl_crop_inside_frac', 0.0) * 100:.0f}%  "
            f"maxR {m.get('pcl_crop_max_dist_cm', 0.0):.1f}cm"
        )

    # summary grid (panels are 224 tall x 896 wide, stacked with 8px gutters)
    ph, pw = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    grid = np.full((rows * (ph + 8), cols * (pw + 8), 3), 24, np.uint8)
    for k, p in enumerate(panels):
        r, c = divmod(k, cols)
        grid[r * (ph + 8) : r * (ph + 8) + ph, c * (pw + 8) : c * (pw + 8) + pw] = p
    gp = out / f"{dataset}_{split}_grid.png"
    cv2.imwrite(str(gp), grid)
    console.print(f"[bold green]grid -> {gp}[/bold green]")


if __name__ == "__main__":
    tyro.cli(main)

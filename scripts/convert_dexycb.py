"""Convert DexYCB to the HUG pkl schema for hand-reconstruction training.

Verified format facts (see analysis_docs/HUG_hand_recon_data_adaptation.md §2):
- <SEQ>/pose.npz: pose_m (N,1,51) = 3 global axis-angle + 45 PCA coefficients
  + 3 translation, WORLD frame (shared by all 8 cameras). All-zero frames have
  no annotation -> skip. (PCA confirmed by the official toolkit's ManoLayer
  with use_pca=True, ncomps=45.)
- PCA -> axis-angle: side-specific `hands_mean + pca @ hands_components` from
  MANO_{LEFT,RIGHT}.pkl.
- Left sequences are canonicalized to the right-hand HUG frame: RGB/depth are
  horizontally mirrored, K is reflected, and global/local rotations are
  conjugated by diag(-1,1,1). `meta.yml['mano_sides'][0]` is authoritative.
- Extrinsics (calibration/extrinsics_<date>/extrinsics.yml) are CAMERA->WORLD
  (3x4); world->camera uses the inverse.
- Intrinsics: calibration/intrinsics/<serial>_640x480.yml (color fx/fy/ppx/ppy).
- Betas: calibration/mano_<date>_<subject>_right/mano.yml (per subject).
- Depth: aligned_depth_to_color_XXXXXX.png, uint16 millimeters (native HUG format).

Usage:
    python scripts/convert_dexycb.py --out-dir /root/code/vepfs/dataset/hand_recon_hug/dexycb \
        [--max-seqs 2] [--workers 16]
"""

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tyro
import yaml
from rich.console import Console

from conversion_common import (
    FrameSample,
    _HUG_ROOT,
    aa_to_rotmat,
    mano_asset,
    mano_pca,
    mano_wrist_offset,
    pose48_to_99d,
    reflect_camera_x,
    reflect_intrinsics,
    reflect_rotation,
    write_sample,
)

console = Console()

DEXYCB_ROOT = Path("/root/code/vepfs/dataset/dex-ycb")
MANO_PKL = _HUG_ROOT / "assets" / "mano" / "models" / "MANO_RIGHT.pkl"

_HANDS_MEAN = None
_HANDS_COMPS = None


def get_pca():
    """Backward-compatible right-hand PCA accessor."""
    return mano_pca("right")


def load_sequence(seq_dir: Path):
    """Load per-sequence constants: meta, pose, per-serial K and world->cam."""
    meta = yaml.unsafe_load(open(seq_dir / "meta.yml"))
    cal = DEXYCB_ROOT / "calibration"
    ext = yaml.unsafe_load(
        open(cal / f"extrinsics_{meta['extrinsics']}" / "extrinsics.yml")
    )["extrinsics"]
    betas = np.asarray(
        yaml.unsafe_load(open(cal / f"mano_{meta['mano_calib'][0]}" / "mano.yml"))["betas"],
        dtype=np.float32,
    )
    pose_m = np.load(seq_dir / "pose.npz")["pose_m"]  # (N,1,51) world frame

    cams = []
    for serial in meta["serials"]:
        intr = yaml.unsafe_load(open(cal / "intrinsics" / f"{serial}_640x480.yml"))
        K = np.array(
            [
                [intr["color"]["fx"], 0, intr["color"]["ppx"]],
                [0, intr["color"]["fy"], intr["color"]["ppy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        T_c2w = np.asarray(ext[serial], dtype=np.float64).reshape(3, 4)  # cam->world
        T_w2c = np.linalg.inv(np.vstack([T_c2w, [0, 0, 0, 1]]))[:3]      # world->cam
        cams.append((serial, K, T_w2c))
    mano_sides = meta.get("mano_sides")
    if not isinstance(mano_sides, list) or len(mano_sides) != 1:
        raise ValueError(f"expected one mano_sides entry in {seq_dir}/meta.yml, got {mano_sides!r}")
    side = str(mano_sides[0]).lower()
    if side not in ("left", "right"):
        raise ValueError(f"unsupported mano side {side!r} in {seq_dir}/meta.yml")
    # Validate the asset before constructing tasks so a missing left asset
    # fails at startup rather than silently corrupting hundreds of frames.
    mano_asset(side)
    return meta, side, pose_m, betas, cams


def _axis_angle_to_rotmat_batch(aa45: np.ndarray) -> list[np.ndarray]:
    return [aa_to_rotmat(aa45[i * 3 : i * 3 + 3]) for i in range(15)]


def _rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].reshape(3).astype(np.float32)


def _canonicalize_left_pose(R_cam, aa45, t_cam):
    """Reflect a left camera-frame pose into the right canonical frame."""
    R_right = reflect_rotation(R_cam)
    aa_right = np.concatenate(
        [_rotmat_to_axis_angle(reflect_rotation(R)) for R in _axis_angle_to_rotmat_batch(aa45)]
    )
    return R_right, aa_right, reflect_camera_x(t_cam)


def _canonicalize_frame_inputs(rgb, depth_mm, K, side):
    if side != "left":
        return rgb, depth_mm, K
    return rgb[:, ::-1].copy(), depth_mm[:, ::-1].copy(), reflect_intrinsics(K, rgb.shape[1])


def convert_frame(task):
    """Convert one source frame to the right-canonical HUG schema."""
    seq_dir, frame_idx, pm, serial, K, T_w2c, betas, side, out_dir = task
    seq_dir = Path(seq_dir)
    try:
        mean, comps = mano_pca(side)
        glob_aa_w = pm[:3].astype(np.float32)
        aa45 = mean + pm[3:48].astype(np.float32) @ comps  # absolute axis-angles
        trans_w = pm[48:51].astype(np.float64)

        # MANO's pose_m translation is the template-origin position. Convert
        # to the true wrist using the source-side template offset, then move
        # world -> camera. HUG's MANO wrapper is wrist-centered.
        R_w = aa_to_rotmat(glob_aa_w)
        R_e, t_e = T_w2c[:, :3], T_w2c[:, 3]
        R_cam = R_e @ R_w
        t_cam = (R_e @ (trans_w + mano_wrist_offset(side)) + t_e).astype(np.float32)

        rgb_path = seq_dir / serial / f"color_{frame_idx:06d}.jpg"
        depth_path = seq_dir / serial / f"aligned_depth_to_color_{frame_idx:06d}.png"
        rgb = cv2.imread(str(rgb_path))
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_mm is None or depth_mm.dtype != np.uint16:
            return None
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        canonicalization = "none"
        if side == "left":
            # Reflect all camera/image quantities into the right canonical
            # frame before shared crop/resize and MANO GT generation.
            R_cam, aa45, t_cam = _canonicalize_left_pose(R_cam, aa45, t_cam)
            rgb, depth_mm, K = _canonicalize_frame_inputs(rgb, depth_mm, K, side)
            canonicalization = "camera_x_reflection"

        glob_aa_cam = _rotmat_to_axis_angle(R_cam)
        pose48_cam = np.concatenate([glob_aa_cam, aa45]).astype(np.float32)
        pose99, R_6d, pose_6d = pose48_to_99d(t_cam, R_cam, aa45)

        rel = seq_dir.relative_to(DEXYCB_ROOT)
        sample = FrameSample(
            stem=f"dexycb_{rel.parts[0]}_{rel.parts[1]}_{serial}_{frame_idx:06d}",
            object_name=f"dexycb_{rel.parts[0]}_{rel.parts[1]}",
            frame_index=int(frame_idx),
            rgb=rgb,
            depth_mm=depth_mm,
            K=K,
            pose99=pose99,
            R_6d=R_6d,
            pose_6d=pose_6d,
            betas_gt=betas,
            pose_aa48=pose48_cam,
            source_mano_side=side,
            canonicalization=canonicalization,
        )
        write_sample(sample, Path(out_dir))
        return sample.stem
    except Exception as e:
        console.print(f"[red]failed {seq_dir.name}/{serial}/{frame_idx}: {e}[/red]")
        return None


def main(
    out_dir: Path,
    max_seqs: int | None = None,
    max_frames_per_seq: int | None = None,
    workers: int = 16,
) -> None:
    seq_dirs = sorted(
        d for d in DEXYCB_ROOT.glob("20*/*/") if (d / "pose.npz").exists()
    )
    if max_seqs is not None:
        seq_dirs = seq_dirs[:max_seqs]
    console.print(f"[cyan]{len(seq_dirs)} sequences found[/cyan]")

    tasks = []
    for seq_dir in seq_dirs:
        meta, side, pose_m, betas, cams = load_sequence(seq_dir)
        n_frames = meta["num_frames"]
        if max_frames_per_seq is not None:
            n_frames = min(n_frames, max_frames_per_seq)
        for frame_idx in range(n_frames):
            pm = pose_m[frame_idx, 0]
            if np.all(pm == 0.0):
                continue
            for serial, K, T_w2c in cams:
                tasks.append(
                    (str(seq_dir), frame_idx, pm, serial, K, T_w2c, betas, side, str(out_dir))
                )
    console.print(f"[cyan]converting {len(tasks)} DexYCB frames -> {out_dir}[/cyan]")

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for stem in ex.map(convert_frame, tasks, chunksize=16):
            done += 1
            if done % 2000 == 0:
                console.print(f"  {done}/{len(tasks)}")
    console.print(f"[green]done: {done} frames processed[/green]")


if __name__ == "__main__":
    tyro.cli(main)

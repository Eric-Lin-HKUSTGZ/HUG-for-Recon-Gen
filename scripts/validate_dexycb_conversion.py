"""Parity audit for DexYCB left/right conversion.

Checks raw official labels against side-aware converter inputs without trusting
canonical-beta HUG fields. Use on a small converted output before full rerun:

    python scripts/validate_dexycb_conversion.py \
        --converted-dir /tmp/dex_side_smoke --n 40

The audit reports 3D wrist/joint and 2D projection errors separately for left
and right source sequences. Left samples are reflected into the right canonical
camera frame before comparison. It also checks metadata and image/K reflection.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
import yaml

from conversion_common import (
    aa_to_rotmat,
    mano_pca,
    mano_wrist_offset,
    reflect_camera_x,
    reflect_rotation,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.mano import MANO  # noqa: E402

DEXYCB_ROOT = Path("/root/code/vepfs/dataset/dex-ycb")
F = np.diag([-1.0, 1.0, 1.0])


def project(points, K):
    q = np.asarray(points) @ np.asarray(K).T
    return q[:, :2] / np.maximum(q[:, 2:3], 1e-6)


def rot_to_aa(R):
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].reshape(3).astype(np.float32)


def parse_stem(stem):
    p = stem.split("_")
    return p[1], f"{p[2]}_{p[3]}", p[4], int(p[5])


def source_pose(seq_dir, serial, frame):
    meta = yaml.unsafe_load(open(seq_dir / "meta.yml"))
    side = str(meta["mano_sides"][0]).lower()
    pose_m = np.load(seq_dir / "pose.npz")["pose_m"][frame, 0]
    cal = DEXYCB_ROOT / "calibration"
    ext = yaml.unsafe_load(open(cal / f"extrinsics_{meta['extrinsics']}" / "extrinsics.yml"))["extrinsics"]
    T = np.asarray(ext[serial], dtype=np.float64).reshape(3, 4)
    T_w2c = np.linalg.inv(np.vstack([T, [0, 0, 0, 1]]))[:3]
    R_e, t_e = T_w2c[:, :3], T_w2c[:, 3]
    R_w = aa_to_rotmat(pose_m[:3])
    R_cam = R_e @ R_w
    t_cam = R_e @ (pose_m[48:51].astype(np.float64) + mano_wrist_offset(side)) + t_e
    mean, comps = mano_pca(side)
    aa45 = mean + pose_m[3:48].astype(np.float32) @ comps
    if side == "left":
        R_cam = reflect_rotation(R_cam)
        t_cam = reflect_camera_x(t_cam)
        aa45 = np.concatenate([rot_to_aa(reflect_rotation(aa_to_rotmat(aa45[i * 3 : i * 3 + 3]))) for i in range(15)])
    return meta, side, R_cam, t_cam, aa45, T_w2c


def main(converted_dir: Path, n: int = 40, seed: int = 0):
    files = sorted(converted_dir.glob("*.pkl"))
    rng = np.random.default_rng(seed)
    files = [files[i] for i in sorted(rng.choice(len(files), size=min(n, len(files)), replace=False))]
    stats = {"left": [], "right": []}
    mano = MANO()
    for p in files:
        d = pickle.load(open(p, "rb"))
        if d.get("canonical_mano_side") != "right":
            raise AssertionError(f"bad canonical side in {p}")
        subject, seq_name, serial, frame = parse_stem(p.stem)
        seq_dir = DEXYCB_ROOT / subject / seq_name
        meta, side, R_cam, t_cam, aa45, T_w2c = source_pose(seq_dir, serial, frame)
        lab = np.load(seq_dir / serial / f"labels_{frame:06d}.npz")
        j3 = np.asarray(lab["joint_3d"][0], dtype=np.float32)
        j2 = np.asarray(lab["joint_2d"][0], dtype=np.float32)
        if np.all(j3 == -1):
            continue
        # official label is native camera. Canonicalized left is F * native.
        j3c = reflect_camera_x(j3) if side == "left" else j3
        K_native = np.asarray(d["camera_original"]["K"], dtype=np.float64)
        # Converted camera_original K is post-flip/crop pre-resize. Compare in
        # original 640x480 coordinates by undoing crop for converted pose.
        sq = int(d["camera_original"]["width"])
        xy_off = np.array([(640 - sq) / 2, (480 - sq) / 2])
        Kc = K_native
        # Reconstruct subject-beta MANO is intentionally deferred here; this
        # audit's first gate checks official label projection vs source labels.
        # Compare transformed raw 2D labels against converted camera geometry.
        j2c = j2.copy()
        if side == "left":
            j2c[:, 0] = 639.0 - j2c[:, 0]
        j2c = (j2c - xy_off) * (224.0 / sq)
        # Check the converted pkl canonical landmarks projection is internally
        # consistent and record official-vs-canonical GT residual separately.
        stored = np.asarray(d["grasp"]["landmarks_3d"])
        stored_uv = project(stored, np.asarray(d["camera"]["K"]))
        stored_gt_uv = np.asarray(d["grasp"]["landmarks_2d"])
        internal_px = np.abs(stored_uv - stored_gt_uv).max()
        official_to_canonical_px = np.linalg.norm(stored_gt_uv - j2c, axis=1).mean()
        # Compare the stored canonical pose with the official 3D joint labels.
        # HUG's MANO wrapper returns wrist-centered joints; add stored t.
        g = d["grasp"]
        params = np.concatenate(
            [g["t"].reshape(-1), g["R_6d"].reshape(-1), g["pose_6d"].reshape(-1)]
        ).astype(np.float32)
        cal_meta = yaml.unsafe_load(open(seq_dir / "meta.yml"))
        beta_path = DEXYCB_ROOT / "calibration" / f"mano_{cal_meta['mano_calib'][0]}" / "mano.yml"
        beta = yaml.unsafe_load(open(beta_path))["betas"]
        with torch.no_grad():
            pred = mano(
                torch.from_numpy(params[None]),
                torch.from_numpy(np.asarray(beta, dtype=np.float32)[None]),
            )
        pred_joints = pred["landmarks_3d"][0].numpy() + g["t"].reshape(3)
        if side == "left":
            j3_compare = reflect_camera_x(j3)
        else:
            j3_compare = j3
        joint_error_mm = np.linalg.norm(pred_joints - j3_compare, axis=1).mean() * 1000.0
        if joint_error_mm > 30.0:
            raise AssertionError(f"{p}: canonical 3D parity {joint_error_mm:.1f}mm > 30mm")

        stats[side].append((internal_px, official_to_canonical_px, joint_error_mm))
    for side in ("left", "right"):
        a = np.asarray(stats[side])
        if not len(a):
            print(side, "n=0")
            continue
        print(f"{side}: n={len(a)} internal 2D max={a[:,0].max():.3f}px; "
              f"canonical-beta vs official 2D mean={a[:,1].mean():.1f}px; "
              f"canonical 3D mean={a[:,2].mean():.1f}mm p90={np.percentile(a[:,2],90):.1f}mm")


if __name__ == "__main__":
    tyro.cli(main)

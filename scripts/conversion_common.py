"""Shared conversion utilities for hand-reconstruction dataset adapters.

Converts per-frame hand MANO annotations into the HUG pkl schema consumed by
`src/dataloader/grasp_dataset.py` (see analysis_docs/HUG_hand_recon_data_adaptation.md).

Conventions (all verified empirically, see the adaptation doc):
- Camera frame, OpenCV convention (z positive forward), metric meters.
- 99D grasp state: t(3) + R_6d(6) + pose_6d(15*6), continuous 6D rotations.
- GT landmarks/mesh are generated with the canonical HUG MANO shape β
  (assets/mano_rhand_shape.npy), matching GraspFlowModel._build_dicts which
  evaluates both pred and GT with fixed_betas (adaptation doc §4.3 option A).
- Images: center-crop square -> resize 224x224, K adjusted (same as
  src/prepare_inputs.py), depth stored as uint16 PNG in 1mm units.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TARGET_SIZE = 224

_HUG_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_BETAS_FILE = _HUG_ROOT / "assets" / "mano_rhand_shape.npy"
MANO_FACES_FILE = _HUG_ROOT / "assets" / "mano_rhand_mesh_faces.npy"
MANO_RIGHT_PKL = _HUG_ROOT / "assets" / "mano" / "models" / "MANO_RIGHT.pkl"
MANO_LEFT_PKL = Path("/root/code/vepfs/GPGFormer/weights/mano/MANO_LEFT.pkl")

_J0_TEMPLATES = {}
_PCA = {}


def _mano_asset(side: str) -> Path:
    if side == "right":
        return MANO_RIGHT_PKL
    if side == "left":
        return MANO_LEFT_PKL
    raise ValueError(f"unsupported MANO side: {side!r}; expected 'left' or 'right'")


def mano_asset(side: str) -> dict:
    """Load one side's MANO asset, failing clearly if it is unavailable."""
    path = _mano_asset(side)
    if not path.is_file():
        raise FileNotFoundError(f"MANO_{side.upper()} asset not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def mano_wrist_offset(side: str = "right") -> np.ndarray:
    """Return J0 = J_regressor @ v_template for the requested hand side.

    DexYCB pose_m/HO3D handTrans identifies the MANO template origin; the
    official wrist is translation + this side-specific, unrotated J0. HUG's
    model uses center_idx=0, so converted 99D ``t`` stores the true wrist.
    """
    if side not in _J0_TEMPLATES:
        d = mano_asset(side)
        Jr = d["J_regressor"]
        Jr = Jr.toarray() if hasattr(Jr, "toarray") else Jr
        _J0_TEMPLATES[side] = (Jr @ d["v_template"])[0].astype(np.float64)
    return _J0_TEMPLATES[side]


def mano_pca(side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the side-specific (hands_mean, hands_components) PCA basis."""
    if side not in _PCA:
        d = mano_asset(side)
        _PCA[side] = (
            np.asarray(d["hands_mean"], dtype=np.float32).reshape(45),
            np.asarray(d["hands_components"], dtype=np.float32).reshape(45, 45),
        )
    return _PCA[side]


def reflect_camera_x(points: np.ndarray) -> np.ndarray:
    """Reflect camera/image X: (x,y,z) -> (-x,y,z)."""
    out = np.asarray(points).copy()
    out[..., 0] *= -1
    return out


def reflect_rotation(R: np.ndarray) -> np.ndarray:
    """Conjugate a proper rotation by the camera-X reflection."""
    F = np.diag([-1.0, 1.0, 1.0])
    return F @ np.asarray(R) @ F


def reflect_intrinsics(K: np.ndarray, width: int) -> np.ndarray:
    """Intrinsics after a horizontal image flip of a ``width``-pixel image."""
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 2] = width - 1 - out[0, 2]
    return out


# --------------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------------

def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    return cv2.Rodrigues(np.asarray(aa, dtype=np.float64).reshape(-1, 3))[0]


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> continuous 6D.

    Layout must round-trip through the model-side decoder
    src/utils/transform_utils.py:six_d_to_rotation_matrix, which does
    six_d.reshape(3, 2) (row-major) and Gram-Schmidts the two COLUMNS.
    The flat form is therefore R[:, :2] in C (row-major) order:
    [R00, R01, R10, R11, R20, R21]. The previous layout
    concat([R[:,0], R[:,1]]) did NOT round-trip: the decoder interleaved the
    two columns and scrambled every stored rotation (~70 deg constant
    orientation error vs official joint_3d, PA residual ~21mm).
    """
    R = np.asarray(R, dtype=np.float32)
    return R[:, :2].reshape(-1).astype(np.float32)


def pose48_to_99d(t_cam: np.ndarray, glob_R: np.ndarray, joint_aa45: np.ndarray):
    """Assemble the HUG 99D state from camera-frame components.

    Args:
        t_cam: (3,) wrist translation in camera frame, meters.
        glob_R: (3,3) global wrist rotation in camera frame.
        joint_aa45: (45,) absolute axis-angles of the 15 MANO finger joints.

    Returns:
        pose99: (99,) float32 = t(3) + R_6d(6) + pose_6d(90).
        R_6d: (6,), pose_6d: (15, 6) for the grasp dict.
    """
    R_6d = rotmat_to_6d(glob_R)
    pose_6d = np.stack(
        [rotmat_to_6d(aa_to_rotmat(joint_aa45[i * 3 : i * 3 + 3])) for i in range(15)]
    ).astype(np.float32)  # (15, 6)
    pose99 = np.concatenate(
        [np.asarray(t_cam, np.float32).reshape(3), R_6d, pose_6d.reshape(-1)]
    ).astype(np.float32)
    return pose99, R_6d, pose_6d


# --------------------------------------------------------------------------
# MANO (canonical-shape GT generation)
# --------------------------------------------------------------------------

_MANO = None
_FACES = None
_CANONICAL_BETAS = None


def get_mano():
    """Process-global MANO instance (safe under multiprocessing workers)."""
    global _MANO, _FACES, _CANONICAL_BETAS
    if _MANO is None:
        import sys

        sys.path.insert(0, str(_HUG_ROOT))
        import torch
        from src.models.mano import MANO

        _MANO = MANO()
        _FACES = np.load(MANO_FACES_FILE)
        _CANONICAL_BETAS = np.load(CANONICAL_BETAS_FILE).astype(np.float32).reshape(1, 10)
    return _MANO, _FACES, _CANONICAL_BETAS


def mano_forward_canonical(pose99: np.ndarray, betas_override: np.ndarray = None):
    """Forward MANO with the canonical HUG shape (or an override β).

    Returns (landmarks_3d_cam (21,3), verts_cam (778,3)) in camera frame.
    """
    import torch

    mano, faces, betas = get_mano()
    if betas_override is not None:
        betas = np.asarray(betas_override, dtype=np.float32).reshape(1, 10)
    with torch.no_grad():
        out = mano(
            torch.from_numpy(pose99.reshape(1, 99).astype(np.float32)),
            betas=torch.from_numpy(betas),
        )
    joints = out["landmarks_3d"][0].cpu().numpy()
    verts = out["vertices"][0].cpu().numpy()
    t = out["t"][0].cpu().numpy()
    return (joints + t).astype(np.float32), (verts + t).astype(np.float32)


# --------------------------------------------------------------------------
# Image packing (mirrors src/prepare_inputs.py)
# --------------------------------------------------------------------------

def center_crop_square(img: np.ndarray):
    h, w = img.shape[:2]
    size = min(h, w)
    x_off, y_off = (w - size) // 2, (h - size) // 2
    return img[y_off : y_off + size, x_off : x_off + size], x_off, y_off


def adjust_K(K: np.ndarray, x_off: int, y_off: int, scale: float) -> np.ndarray:
    K_new = np.asarray(K, dtype=np.float64).copy()
    K_new[0, 2] -= x_off
    K_new[1, 2] -= y_off
    K_new[:2, :] *= scale
    return K_new


def project_uv(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,3) camera-frame points -> (N,2) pixels."""
    z = np.maximum(xyz[:, 2:3], 1e-6)
    uv = xyz[:, :2] / z
    return np.stack(
        [uv[:, 0] * K[0, 0] + K[0, 2], uv[:, 1] * K[1, 1] + K[1, 2]], axis=-1
    ).astype(np.float32)


def make_hand_mask(verts_2d: np.ndarray, img_size: int = TARGET_SIZE) -> np.ndarray:
    """Convex hull of projected mesh verts, dilated; uint8 (H,W) mask."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    pts = np.round(verts_2d).astype(np.int32)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < img_size) & (pts[:, 1] >= 0) & (pts[:, 1] < img_size)]
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return mask


@dataclass
class FrameSample:
    """Everything needed to write one HUG-schema pkl."""

    stem: str
    object_name: str
    frame_index: int
    rgb: np.ndarray            # (H,W,3) uint8 RGB
    depth_mm: np.ndarray       # (H,W) uint16, 1mm units, same HxW as rgb
    K: np.ndarray              # (3,3) at the rgb resolution
    pose99: np.ndarray         # (99,) camera-frame MANO state
    R_6d: np.ndarray           # (6,)
    pose_6d: np.ndarray        # (15,6)
    betas_gt: np.ndarray       # (10,) dataset-provided shape (kept for future use)
    pose_aa48: np.ndarray      # (48,) axis-angle (global + 15 joints), camera frame
    source_mano_side: str = "right"
    canonical_mano_side: str = "right"
    canonicalization: str = "none"
    hand_mask: np.ndarray | None = None  # optional source-resolution hand mask


def write_sample(sample: FrameSample, out_dir: Path) -> Path:
    """Convert one frame to the HUG pkl schema and write it."""
    mano, faces, canonical_betas = get_mano()

    # --- crop square / resize / K adjust (224) ---
    rgb_sq, x_off, y_off = center_crop_square(sample.rgb)
    sq = rgb_sq.shape[0]
    scale = TARGET_SIZE / sq
    K_orig = adjust_K(sample.K, x_off, y_off, 1.0)
    K_224 = adjust_K(sample.K, x_off, y_off, scale)

    rgb_224 = cv2.resize(rgb_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    depth_sq, _, _ = center_crop_square(sample.depth_mm)
    depth_224 = cv2.resize(depth_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

    # --- canonical-shape MANO forward (GT landmarks/mesh) ---
    joints_cam, verts_cam = mano_forward_canonical(sample.pose99)
    verts_2d = project_uv(verts_cam, K_224)
    joints_2d = project_uv(joints_cam, K_224)
    # Mask from the SUBJECT-shaped mesh (betas_gt): the canonical HUG hand is
    # smaller than many subjects' hands, and the mask's job is to cover the
    # true hand pixels for query-point sampling.
    _, verts_subj = mano_forward_canonical(sample.pose99, betas_override=sample.betas_gt)
    mask = make_hand_mask(project_uv(verts_subj, K_224))

    # --- encodings ---
    _, img_buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR))
    _, depth_buf = cv2.imencode(".png", depth_224)
    _, mask_buf = cv2.imencode(".png", mask)

    t = sample.pose99[:3].reshape(1, 3)
    T_camera_wrist = np.eye(4, dtype=np.float32)
    T_camera_wrist[:3, :3] = aa_to_rotmat(sample.pose_aa48[:3])
    T_camera_wrist[:3, 3] = t.reshape(3)

    entry = {
        "object_name": sample.object_name,
        "frame_index": int(sample.frame_index),
        "grasp_index": 0,
        "camera": {"K": K_224, "width": TARGET_SIZE, "height": TARGET_SIZE},
        "camera_original": {"K": K_orig, "width": sq, "height": sq},
        "grasp": {
            "pose": np.asarray(sample.pose_aa48[3:], np.float32).reshape(1, 15, 3),
            "pose_6d": np.asarray(sample.pose_6d, np.float32).reshape(1, 15, 6),
            "shape": canonical_betas.reshape(1, 10),          # canonical (HUG 约定)
            "shape_gt": np.asarray(sample.betas_gt, np.float32).reshape(1, 10),
            "landmarks_3d": joints_cam.astype(np.float32),
            "landmarks_2d": joints_2d.astype(np.float32),
            "T_camera_wrist": T_camera_wrist,
            "R_6d": np.asarray(sample.R_6d, np.float32).reshape(1, 6),
            "t": t.astype(np.float32),
            "mesh_vertices": verts_cam.astype(np.float32),
            "mesh_faces": faces,
        },
        "image": img_buf.tobytes(),
        "depth": depth_buf.tobytes(),
        "object_mask": mask_buf.tobytes(),
        "schema_version": "dexycb_hug_v2",
        "source_mano_side": sample.source_mano_side,
        "canonical_mano_side": sample.canonical_mano_side,
        "canonicalization": sample.canonicalization,
        # 不写 condition_point：让 GraspDataset 训练时从手部 mask 随机采样 query 点
    }

    out_path = out_dir / f"{sample.stem}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(entry, f)
    tmp.rename(out_path)
    return out_path

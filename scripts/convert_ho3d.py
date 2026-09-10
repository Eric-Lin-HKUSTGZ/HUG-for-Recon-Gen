"""Convert HO3D_v3 to the HUG pkl schema for hand-reconstruction training.

Verified format facts (see analysis_docs/HUG_hand_recon_data_adaptation.md §2):
- meta/<FRAME>.pkl: camMat (3,3), handPose (48 axis-angle), handBeta (10),
  handTrans (3), handJoints3D (21,3). Some frames have empty handJoints3D -> skip.
- Coordinate convention: z negative in front, y flipped. Convert with
  (x,y,z) -> (x,-y,-z); global orient: Rconv @ R with Rconv = diag(1,-1,-1).
- depth/<FRAME>.png: 3-channel uint8 encoding a 16-bit value with B=0,
  G=high byte, R=low byte: depth_m = (G*256 + R) * 0.00012498664727900177.

Splits:
- --split train (default): reads train.txt, full MANO annotations -> training pkls.
- --split evaluation: reads evaluation.txt. The eval set has NO MANO params
  (verified: handPose/handBeta/handTrans are empty in eval meta pkls); the
  official GT lives in evaluation_xyz.json (21 joints) / evaluation_verts.json
  (mesh verts). We therefore write eval-schema pkls (grasp=None) carrying the
  input image/depth/camera plus joints_gt/verts_gt for metric computation, and
  set condition_point to the projected wrist so inference can run directly.
  joints_gt is stored verbatim in the OFFICIAL raw MANO kinematic order
  [wrist, idx x3, mid x3, pinky x3, ring x3, thumb x3, tips(thumb,idx,mid,ring,
  pinky)] - GraspDataset reorders it to our standard order at load time
  (HO3D_RAW_TO_STD in src/dataloader/grasp_dataset.py). Do NOT permute here.

Usage:
    python scripts/convert_ho3d.py --out-dir .../ho3d [--max-samples 200] [--workers 16]
    python scripts/convert_ho3d.py --split evaluation --out-dir .../ho3d_eval
"""

import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tyro
from rich.console import Console

from conversion_common import (
    TARGET_SIZE,
    adjust_K,
    center_crop_square,
    project_uv,
    FrameSample,
    aa_to_rotmat,
    mano_wrist_offset,
    pose48_to_99d,
    write_sample,
)

console = Console()

HO3D_ROOT = Path("/root/code/vepfs/dataset/HO3D_v3")
DEPTH_SCALE = 0.00012498664727900177
RCONV = np.diag([1.0, -1.0, -1.0])  # HO3D cam -> OpenCV cam (180 deg about x)


def decode_depth_m(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None or d.ndim != 3:
        raise ValueError(f"bad depth image: {path}")
    raw16 = d[:, :, 1].astype(np.uint32) * 256 + d[:, :, 2].astype(np.uint32)
    return raw16 * DEPTH_SCALE  # meters, float32


def ho3d_to_std(xyz: np.ndarray) -> np.ndarray:
    """HO3D camera coords -> OpenCV convention: (x,y,z) -> (x,-y,-z)."""
    return np.asarray(xyz, dtype=np.float32) * np.array([1.0, -1.0, -1.0], np.float32)


def write_eval_sample(task):
    """Worker: one evaluation frame -> eval-schema pkl (grasp=None + GT joints/verts)."""
    seq, frame_id, joints_gt, verts_gt, out_dir = task
    seq_dir = HO3D_ROOT / "evaluation" / seq
    rgb_path = seq_dir / "rgb" / f"{frame_id}.jpg"
    meta_path = seq_dir / "meta" / f"{frame_id}.pkl"
    depth_path = seq_dir / "depth" / f"{frame_id}.png"
    if not (rgb_path.exists() and meta_path.exists() and depth_path.exists()):
        return None
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        K = np.asarray(meta["camMat"], dtype=np.float64)

        joints_cam = ho3d_to_std(np.asarray(joints_gt, np.float32).reshape(21, 3))
        verts_cam = (
            ho3d_to_std(np.asarray(verts_gt, np.float32).reshape(-1, 3))
            if verts_gt is not None
            else None
        )

        # crop square / resize 224 / K adjust（与训练 pkl 同款处理）
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        rgb_sq, x_off, y_off = center_crop_square(rgb)
        sq = rgb_sq.shape[0]
        scale = TARGET_SIZE / sq
        K_orig = adjust_K(K, x_off, y_off, 1.0)
        K_224 = adjust_K(K, x_off, y_off, scale)
        rgb_224 = cv2.resize(rgb_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        depth_mm = np.clip(decode_depth_m(depth_path) * 1000.0, 0, 65535).astype(np.uint16)
        depth_sq, _, _ = center_crop_square(depth_mm)
        depth_224 = cv2.resize(depth_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

        joints_2d = project_uv(joints_cam, K_224)
        wrist_uv = joints_2d[0]  # query 点 = 手腕投影（评估时定位用）

        _, img_buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR))
        _, depth_buf = cv2.imencode(".png", depth_224)

        entry = {
            "object_name": f"ho3d_eval_{seq}",
            "frame_index": int(frame_id),
            "grasp_index": 0,
            "camera": {"K": K_224, "width": TARGET_SIZE, "height": TARGET_SIZE},
            "camera_original": {"K": K_orig, "width": sq, "height": sq},
            "grasp": None,
            "image": img_buf.tobytes(),
            "depth": depth_buf.tobytes(),
            "object_mask": b"",
            "condition_point": wrist_uv.astype(np.float32),
            "joints_gt": joints_cam.astype(np.float32),   # (21,3) 相机系米制，OpenCV 约定
            "verts_gt": (
                verts_cam.astype(np.float32) if verts_cam is not None else None
            ),
        }
        stem = f"ho3d_eval_{seq}_{frame_id}"
        out_path = Path(out_dir) / f"{stem}.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.pkl")
        with open(tmp, "wb") as f:
            pickle.dump(entry, f)
        tmp.rename(out_path)
        return stem
    except Exception as e:
        console.print(f"[red]failed eval {seq}/{frame_id}: {e}[/red]")
        return None


def convert_frame(task):
    """Worker: one (seq, frame_id) -> one pkl. Returns stem or None."""
    seq, frame_id, out_dir = task
    meta_path = HO3D_ROOT / "train" / seq / "meta" / f"{frame_id}.pkl"
    rgb_path = HO3D_ROOT / "train" / seq / "rgb" / f"{frame_id}.jpg"
    depth_path = HO3D_ROOT / "train" / seq / "depth" / f"{frame_id}.png"
    if not (meta_path.exists() and rgb_path.exists() and depth_path.exists()):
        return None
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        joints = np.asarray(meta.get("handJoints3D"), dtype=np.float32)
        if joints.shape != (21, 3):  # empty annotation
            return None

        K = np.asarray(meta["camMat"], dtype=np.float64)
        pose48 = np.asarray(meta["handPose"], dtype=np.float32).reshape(48)
        betas = np.asarray(meta["handBeta"], dtype=np.float32).reshape(10)
        trans = np.asarray(meta["handTrans"], dtype=np.float32).reshape(3)

        # HO3D convention -> OpenCV: flip y,z for translations; Rconv @ R for global orient.
        # handTrans is the MANO template-ORIGIN position, not the wrist:
        # official handJoints3D wrist = handTrans + J0_template (conversion_common).
        t_cam = ho3d_to_std(trans + mano_wrist_offset())
        R_glob = RCONV @ aa_to_rotmat(pose48[:3])
        glob_aa_cam = cv2.Rodrigues(R_glob)[0].reshape(3).astype(np.float32)
        pose48_cam = np.concatenate([glob_aa_cam, pose48[3:]]).astype(np.float32)

        pose99, R_6d, pose_6d = pose48_to_99d(t_cam, R_glob, pose48[3:])

        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth_mm = np.clip(decode_depth_m(depth_path) * 1000.0, 0, 65535).astype(np.uint16)

        sample = FrameSample(
            stem=f"ho3d_{seq}_{frame_id}",
            object_name=f"ho3d_{seq}",
            frame_index=int(frame_id),
            rgb=rgb,
            depth_mm=depth_mm,
            K=K,
            pose99=pose99,
            R_6d=R_6d,
            pose_6d=pose_6d,
            betas_gt=betas,
            pose_aa48=pose48_cam,
        )
        write_sample(sample, Path(out_dir))
        return sample.stem
    except Exception as e:
        console.print(f"[red]failed {seq}/{frame_id}: {e}[/red]")
        return None


def main(
    out_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
    workers: int = 16,
) -> None:
    if split == "evaluation":
        lines = [
            l.strip()
            for l in (HO3D_ROOT / "evaluation.txt").read_text().splitlines()
            if l.strip()
        ]
        eval_xyz = json.load(open(HO3D_ROOT / "evaluation_xyz.json"))
        eval_verts = json.load(open(HO3D_ROOT / "evaluation_verts.json"))
        assert len(lines) == len(eval_xyz) == len(eval_verts), "eval GT 长度不匹配"
        tasks = []
        for line, xyz, verts in zip(lines, eval_xyz, eval_verts):
            seq, frame_id = line.split("/")
            tasks.append((seq, frame_id, xyz, verts, str(out_dir)))
        if max_samples is not None:
            tasks = tasks[:max_samples]
        console.print(f"[cyan]converting {len(tasks)} HO3D eval frames -> {out_dir}[/cyan]")
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for stem in ex.map(write_eval_sample, tasks, chunksize=32):
                done += 1
                if done % 2000 == 0:
                    console.print(f"  {done}/{len(tasks)}")
        console.print(f"[green]done: {done} eval frames processed[/green]")
        return

    lines = [l.strip() for l in (HO3D_ROOT / "train.txt").read_text().splitlines() if l.strip()]
    tasks = []
    for line in lines:
        seq, frame_id = line.split("/")
        tasks.append((seq, frame_id, str(out_dir)))
    if max_samples is not None:
        tasks = tasks[:max_samples]
    console.print(f"[cyan]converting {len(tasks)} HO3D frames -> {out_dir}[/cyan]")

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for stem in ex.map(convert_frame, tasks, chunksize=32):
            done += 1
            if done % 2000 == 0:
                console.print(f"  {done}/{len(tasks)}")
    console.print(f"[green]done: {done} frames processed[/green]")


if __name__ == "__main__":
    tyro.cli(main)

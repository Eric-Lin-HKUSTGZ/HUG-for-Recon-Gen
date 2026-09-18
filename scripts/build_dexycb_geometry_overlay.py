"""Build compact native DexYCB corrections without rewriting any image PKLs.

Reads official camera-frame labels and subject beta. Left RGB was already
mirrored in v4 PKLs: only native labels are mirrored here, exactly once.
Every frame is validated against official joints before becoming usable.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.dataloader.geometry_overlay import ARRAYS, CONVENTION, VERSION
from src.models.native_mano import NativeCanonicalMANO
from scripts.conversion_common import mano_pca

STEM = re.compile(r"dexycb_(\d{8}-subject-\d{2})_(\d{8}_\d{6})_(\d+)_(\d{6})")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".overlay-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_stems(path):
    names = [v.strip().removesuffix(".pkl") for v in Path(path).read_text().splitlines() if v.strip()]
    if len(set(names)) != len(names) or any(STEM.fullmatch(v) is None for v in names):
        raise ValueError(f"invalid or duplicate DexYCB sample names: {path}")
    return names


class NativeLabels:
    def __init__(self, raw_root, left_asset):
        self.raw_root = Path(raw_root).resolve()
        # Decode PCA using the actual asset selected for each side.
        self.basis = {"right": mano_pca("right")}
        import pickle
        with Path(left_asset).open("rb") as f:
            left = pickle.load(f, encoding="latin1")
        self.basis["left"] = (np.asarray(left["hands_mean"], np.float32).reshape(45),
                              np.asarray(left["hands_components"], np.float32).reshape(45, 45))

    @lru_cache(maxsize=2000)
    def sequence(self, subject, sequence):
        path = self.raw_root / subject / sequence / "meta.yml"
        meta = yaml.safe_load(path.read_text())
        if len(meta["mano_sides"]) != 1 or meta["mano_sides"][0] not in ("left", "right"):
            raise ValueError(f"invalid hand-side annotation: {path}")
        beta_path = self.raw_root / "calibration" / ("mano_" + meta["mano_calib"][0]) / "mano.yml"
        beta = np.asarray(yaml.safe_load(beta_path.read_text())["betas"], np.float32).reshape(10)
        return meta["mano_sides"][0], beta, hashlib.sha256(path.read_bytes() + beta_path.read_bytes()).hexdigest()

    @lru_cache(maxsize=16)
    def camera(self, serial):
        c = yaml.full_load((self.raw_root / "calibration/intrinsics" / f"{serial}_640x480.yml").read_text())["color"]
        return np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], np.float32)

    def read(self, stem):
        subject, sequence, serial, frame = STEM.fullmatch(stem).groups()
        side, beta, metadata_hash = self.sequence(subject, sequence)
        path = self.raw_root / subject / sequence / serial / f"labels_{frame}.npz"
        # Small labels only: no RGB, depth or existing large PKL is opened.
        payload = path.read_bytes()
        import io
        with np.load(io.BytesIO(payload), allow_pickle=False) as labels:
            pm = np.asarray(labels["pose_m"], np.float32).reshape(51)
            joints = np.asarray(labels["joint_3d"], np.float32).reshape(21, 3).copy()
            uv = np.asarray(labels["joint_2d"], np.float32).reshape(21, 2).copy()
        mean, comps = self.basis[side]
        aa = np.concatenate((pm[:3], mean + pm[3:48] @ comps)).reshape(16, 3)
        k = self.camera(serial).copy()
        left = side == "left"
        if left:
            aa = aa * np.array([1, -1, -1], np.float32)
            joints[:, 0] *= -1
            uv[:, 0] = 639 - uv[:, 0]
            k[0, 2] = 639 - k[0, 2]
        matrices = Rotation.from_rotvec(aa).as_matrix().astype(np.float32)
        params = np.concatenate((joints[0], matrices[:, :, :2].reshape(-1), beta)).astype(np.float32)
        if not all(np.isfinite(v).all() for v in (params, joints, uv)) or np.any(joints[:, 2] <= 0):
            raise ValueError(f"invalid official label: {stem}")
        proj = joints.astype(np.float64) @ k.astype(np.float64).T
        uv_error = float(np.abs(proj[:, :2] / proj[:, 2:] - uv).max())
        if uv_error > 0.01:
            raise ValueError(f"official 2D/3D projection mismatch {uv_error} px: {stem}")
        fingerprint = hashlib.sha256(payload + metadata_hash.encode()).hexdigest()
        return params, aa.reshape(48), joints, k, left, uv_error, fingerprint


def create_arrays(directory, names, resume):
    n = len(names)
    arrays = {}
    for key, (shape, dtype) in ARRAYS.items():
        path = directory / f"{key}.npy"
        if resume:
            value = np.load(path, mmap_mode="r+", allow_pickle=False)
            if value.shape != (n, *shape) or value.dtype != dtype:
                raise ValueError(f"resume array mismatch: {key}")
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n, *shape))
        arrays[key] = value
    if not resume:
        np.save(directory / "samples.npy", np.asarray(names))
        hashes = np.lib.format.open_memmap(directory / "source_sha256.npy", mode="w+", dtype="S64", shape=(n,))
    else:
        if not np.array_equal(np.load(directory / "samples.npy", allow_pickle=False), names):
            raise ValueError("resume sample list mismatch")
        hashes = np.load(directory / "source_sha256.npy", mmap_mode="r+", allow_pickle=False)
        if hashes.shape != (n,) or hashes.dtype != np.dtype("S64"):
            raise ValueError("resume source fingerprints mismatch")
    arrays["source_sha256"] = hashes
    return arrays


def training_stats(params, names, train_set):
    n, total, square = 0, np.zeros(109, np.float64), np.zeros(109, np.float64)
    for start in range(0, len(names), 8192):
        selected = [i for i in range(start, min(start + 8192, len(names))) if names[i] in train_set]
        if not selected:
            continue
        x = np.asarray(params[selected], np.float64)
        total += x.sum(0)
        square += (x * x).sum(0)
        n += len(x)
    if not n:
        raise ValueError("selection contains no training frames for normalization")
    mean = total / n
    std = np.sqrt(np.maximum(square / n - mean * mean, 0))
    std = np.where(std > 1e-6, std, 1.0)
    stats = {}
    for key, lo, hi in (("translation", 0, 3), ("wrist_rot", 3, 9), ("finger_rot", 9, 99), ("shape", 99, 109)):
        stats[key] = {"mean": mean[lo:hi].tolist(), "std": std[lo:hi].tolist()}
    return stats, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("/root/code/vepfs/dataset/dex-ycb"))
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--val-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--left-asset", type=Path, default=Path("/root/code/vepfs/GPGFormer/weights/mano/MANO_LEFT.pkl"))
    p.add_argument("--per-split", type=int, default=0, help="0=all; positive=evenly spaced smoke subset")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.batch_size, args.workers, args.torch_threads) < 1 or args.per_split < 0:
        p.error("invalid batch size, threads, workers or per-split count")
    out = args.out_dir.resolve()
    for source in (args.dataset_root.resolve(), args.raw_root.resolve()):
        if not source.is_dir() or source == out or source in out.parents or out in source.parents:
            p.error("output must be separate from the existing dataset directories")
    splits = {k: read_stems(getattr(args, f"{k}_list")) for k in ("train", "val", "test")}
    if sum(map(len, splits.values())) != len(set().union(*map(set, splits.values()))):
        raise ValueError("sample-name overlap between data splits")
    names = sorted(v for values in splits.values() for v in
                   (values if args.per_split == 0 else [values[i] for i in np.linspace(0, len(values)-1, min(len(values), args.per_split), dtype=int)]))
    if not names:
        raise ValueError("empty sample selection")
    signature_paths = [Path(__file__), REPO / "src/models/native_mano.py", REPO / "src/models/mano.py",
                       REPO / "src/dataloader/geometry_overlay.py", REPO / "src/utils/transform_utils.py",
                       REPO / "assets/mano/models/MANO_RIGHT.pkl", args.left_asset]
    signature = {str(v.resolve()): sha256(v) for v in signature_paths}
    manifest = {"version": VERSION, "parameter_convention": CONVENTION,
                "dataset_root": str(args.dataset_root.resolve()), "raw_root": str(args.raw_root.resolve()),
                "count": len(names), "per_split": args.per_split, "signature": signature,
                "selection_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                "split_sha256": {k: sha256(getattr(args, f"{k}_list")) for k in splits}}
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive builder lock; a stale lock requires an explicit human check.
    lock = out / ".builder.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        resume = (out / "manifest.json").exists()
        if resume:
            old = json.loads((out / "manifest.json").read_text())
            if any(old.get(k) != v for k, v in manifest.items()):
                raise ValueError("existing overlay manifest differs; use another output directory")
            if old.get("status") == "complete":
                for filename, expected in old.get("array_sha256", {}).items():
                    if sha256(out / filename) != expected:
                        raise ValueError(f"completed overlay is corrupted: {filename}")
                if sha256(out / "norm_stats.json") != old["norm_stats_sha256"]:
                    raise ValueError("completed overlay normalization is corrupted")
                print("Already complete; no output modified", flush=True)
                return
            if not args.resume:
                raise ValueError("incomplete overlay exists; use --resume")
        elif any(v.name != ".builder.lock" for v in out.iterdir()):
            raise ValueError("output is nonempty without a matching manifest")
        torch.set_num_threads(args.torch_threads)
        model = NativeCanonicalMANO(args.left_asset).eval()
        labels = NativeLabels(args.raw_root, args.left_asset)
        arrays = create_arrays(out, names, resume)
        if not resume:
            np.save(out / "faces_right.npy", model.mano_layer.th_faces.numpy().astype(np.int32))
            np.save(out / "faces_left.npy", model.left_layer.th_faces.numpy()[:, [0, 2, 1]].astype(np.int32))
            atomic_json(out / "manifest.json", dict(manifest, status="running"))
        progress_path = out / "progress.json"
        progress = json.loads(progress_path.read_text()) if resume and progress_path.exists() else {"processed": 0, "max_joint_error_mm": 0.0, "max_projection_error_px": 0.0}
        start = int(progress["processed"])
        if start < 0 or start > len(names):
            raise ValueError("invalid resume progress")
        # Revalidate original small labels before trusting completed chunks.
        if resume:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for offset in range(0, start, args.batch_size):
                    end = min(offset + args.batch_size, start)
                    records = list(pool.map(labels.read, names[offset:end]))
                    actual = np.asarray([v[6] for v in records], dtype="S64")
                    if not np.array_equal(arrays["source_sha256"][offset:end], actual):
                        raise ValueError("raw labels changed since interrupted build")
        begin = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool, torch.inference_mode():
            for offset in range(start, len(names), args.batch_size):
                end = min(offset + args.batch_size, len(names))
                records = list(pool.map(labels.read, names[offset:end]))
                x = torch.from_numpy(np.stack([r[0] for r in records]))
                left = torch.tensor([r[4] for r in records], dtype=torch.bool)
                truth = np.stack([r[2] for r in records])
                pred = model(x, source_is_left=left)
                joints = (pred["landmarks_3d"] + x[:, None, :3]).numpy()
                vertices = (pred["vertices"] + x[:, None, :3]).numpy()
                error = np.linalg.norm(joints.astype(np.float64) - truth, axis=-1) * 1000
                if not np.isfinite(vertices).all() or error.max() > 0.05:
                    raise ValueError(f"native decode disagrees with official joints at {names[offset]}: {error.max()} mm")
                arrays["params"][offset:end] = x.numpy()
                arrays["pose_aa"][offset:end] = np.stack([r[1] for r in records])
                arrays["joints"][offset:end] = truth  # official GT, never overwritten by model output
                arrays["vertices"][offset:end] = vertices
                arrays["camera_K"][offset:end] = np.stack([r[3] for r in records])
                arrays["source_is_left"][offset:end] = left.numpy()
                arrays["source_sha256"][offset:end] = np.asarray([r[6] for r in records], dtype="S64")
                for value in arrays.values():
                    value.flush()
                progress.update(processed=end, max_joint_error_mm=max(progress["max_joint_error_mm"], float(error.max())),
                                max_projection_error_px=max(progress["max_projection_error_px"], max(r[5] for r in records)))
                atomic_json(progress_path, progress)
                if offset == start or end == len(names) or (offset // args.batch_size) % 10 == 0:
                    print(json.dumps(dict(progress, total=len(names), elapsed_seconds=round(time.monotonic()-begin, 1))), flush=True)
        stats, train_count = training_stats(arrays["params"], names, set(splits["train"]))
        atomic_json(out / "norm_stats.json", stats)
        selected = set(names)
        for split, values in splits.items():
            (out / f"{split}.txt").write_text("\n".join(v for v in values if v in selected) + "\n")
        hashes = {v.name: sha256(v) for v in sorted(out.glob("*.npy"))}
        atomic_json(out / "manifest.json", dict(manifest, status="complete", train_count=train_count,
                    validation=progress, array_sha256=hashes, norm_stats_sha256=sha256(out / "norm_stats.json")))
        print(json.dumps({"status": "complete", "count": len(names), "train_count": train_count, "out_dir": str(out)}), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

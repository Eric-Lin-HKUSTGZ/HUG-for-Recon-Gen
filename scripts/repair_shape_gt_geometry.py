"""修复重建 pkl：用 shape_gt 重新生成几何字段，另存到新目录。

只更新 grasp.landmarks_2d / landmarks_3d / mesh_vertices，并增加修复溯源。
不修改 RGB、Depth、mask、相机、姿态、shape 或 shape_gt。原目录永不写入。
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.models.mano import MANO

VERSION = "shape_gt_geometry_v1"
MARKER = "shape_gt_geometry_repair"


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def params109(data):
    """与 GraspDataset._get_mano_params 的 109D 路径一致，要求真实 shape_gt。"""
    g = data["grasp"]
    parts = []
    for key, width in [("t", 3), ("R_6d", 6), ("pose_6d", 90), ("shape_gt", 10)]:
        part = np.asarray(g[key], dtype=np.float32).reshape(-1)
        if part.size != width or not np.isfinite(part).all():
            raise ValueError(f"invalid {key}: expected {width} finite values")
        parts.append(part)
    return np.concatenate(parts)


@torch.inference_mode()
def repair_batch(records, mano, device):
    """返回新的顶层/grasp 字典；不改变传入记录或其他字段。"""
    x = torch.as_tensor(np.stack([params109(d) for d in records]), device=device)
    ks = np.stack([np.asarray(d["camera"]["K"], dtype=np.float32) for d in records])
    if ks.shape != (len(records), 3, 3) or not np.isfinite(ks).all():
        raise ValueError("camera.K must contain finite 3x3 matrices")
    if np.any(np.abs(np.linalg.det(ks)) < 1e-8):
        raise ValueError("camera.K is singular")
    out = mano(x, betas=x[:, 99:109])
    joints = out["landmarks_3d"] + x[:, None, :3]
    vertices = out["vertices"] + x[:, None, :3]
    projection = joints @ torch.as_tensor(ks, device=device).transpose(1, 2)
    if (joints[..., 2] <= 1e-6).any() or (projection[..., 2].abs() <= 1e-6).any():
        raise ValueError("invalid joint depth; refusing to clamp invalid projections")
    uv = projection[..., :2] / projection[..., 2:]
    arrays = [a.cpu().numpy().astype(np.float32) for a in (uv, joints, vertices)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("MANO produced non-finite geometry")
    fixed = []
    for i, record in enumerate(records):
        g = dict(record["grasp"])
        for key, array in zip(("landmarks_2d", "landmarks_3d", "mesh_vertices"), arrays):
            g[key] = array[i].copy()
        fixed.append({**record, "grasp": g})
    return fixed


def implementation_signature():
    """代码和 MANO 资产指纹，防止续跑时混入不同实现的结果。"""
    digest = hashlib.sha256()
    paths = [Path(__file__), REPO / "src/models/mano.py", REPO / "src/utils/transform_utils.py"]
    paths += sorted((REPO / "assets/mano").rglob("*.pkl"))
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_one(source, output, rel, signature):
    if not (source / rel).resolve().is_relative_to(source):
        raise ValueError(f"sample escapes source directory: {rel}")
    raw = (source / rel).read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    expected = {"version": VERSION, "implementation_sha256": signature,
                "source_sha256": source_hash, "shape_source": "grasp.shape_gt"}
    target = output / rel
    if target.exists():
        with target.open("rb") as f:
            existing = pickle.load(f)
        if existing.get(MARKER) != expected:
            raise ValueError(f"existing output does not match source/implementation: {target}; use a new output directory")
        return rel, None, expected
    return rel, pickle.loads(raw), expected


def collect_files(source, lists, max_samples):
    paths = set()
    for filename in lists:
        for line in Path(filename).read_text().splitlines():
            if not line.strip():
                continue
            rel = Path(line.strip())
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"sample path must be relative: {line}")
            if rel.suffix != ".pkl":
                rel = Path(str(rel) + ".pkl")
            paths.add(rel)
    paths = sorted(paths)
    if max_samples and len(paths) > max_samples:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_samples, dtype=int)]
    if not paths:
        raise ValueError("no samples selected")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, nargs="+", required=True,
                        help="一个或多个样本清单；合并去重，保持相对路径")
    parser.add_argument("--max-samples", type=int, default=0, help="0=全量；正数=均匀抽样 smoke test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu（默认，不占训练 GPU）或 cuda:0 等")
    args = parser.parse_args()
    if min(args.batch_size, args.io_workers, args.torch_threads) < 1 or args.max_samples < 0:
        parser.error("batch-size/io-workers/torch-threads must be positive; max-samples >= 0")
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("source-dir and output-dir must be separate, non-nested directories")
    if not source.is_dir():
        parser.error("source-dir does not exist")
    paths = collect_files(source, args.samples, args.max_samples)
    signature = implementation_signature()
    selection_hash = hashlib.sha256("\n".join(p.as_posix() for p in paths).encode()).hexdigest()
    manifest = {"version": VERSION, "source_dir": str(source),
                "implementation_sha256": signature, "selection_sha256": selection_hash,
                "selected": len(paths)}
    manifest_path = output / "repair_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("output manifest differs; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output directory is nonempty and has no repair manifest")
    atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
    torch.set_num_threads(args.torch_threads)
    model = MANO().to(args.device).eval()
    done = skipped = 0
    start = time.monotonic()
    report = {**manifest, "status": "running", "output_dir": str(output)}
    try:
        with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
            for offset in range(0, len(paths), args.batch_size):
                batch = list(pool.map(lambda rel: load_one(source, output, rel, signature),
                                      paths[offset:offset + args.batch_size]))
                pending = [(rel, d, marker) for rel, d, marker in batch if d is not None]
                skipped += len(batch) - len(pending)
                if pending:
                    repaired = repair_batch([d for _, d, _ in pending], model, args.device)
                    writes = []
                    for (rel, _, marker), data in zip(pending, repaired):
                        data[MARKER] = marker
                        writes.append((output / rel, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)))
                    list(pool.map(lambda item: atomic_write(*item), writes))
                    done += len(pending)
                elapsed = time.monotonic() - start
                if offset == 0 or (offset // args.batch_size) % 20 == 0 or done + skipped == len(paths):
                    print(json.dumps({"processed": done + skipped, "total": len(paths),
                                      "written": done, "resumed": skipped,
                                      "samples_per_second": round((done + skipped) / max(elapsed, 1e-6), 2)}), flush=True)
        atomic_write(output / "samples.txt", ("\n".join(p.with_suffix("").as_posix() for p in paths) + "\n").encode())
        report["status"] = "success"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report.update(written_this_run=done, resumed_this_run=skipped,
                      elapsed_seconds=round(time.monotonic() - start, 3))
        atomic_write(output / "repair_report.json", json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

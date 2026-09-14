"""方案 B：复用正确的 MANO 标注，重新读取原始 DexYCB RGBD，生成完整分辨率 pkl。

不做中心裁剪、不 resize；左手仍镜像到右手相机坐标系。几何统一由 shape_gt
生成。RGB/Depth/mask 均无损 PNG 编码；训练时再由 GT/detector 裁 RGB。
"""

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from conversion_common import reflect_intrinsics
from repair_shape_gt_geometry import atomic_write, collect_files, repair_batch
from src.models.mano import MANO

VERSION = "dexycb_hug_v4_fullres_shape_gt"
STEM_RE = re.compile(r"dexycb_(\d{8}-subject-\d{2})_(\d{8}_\d{6})_(\d+)_(\d{6})")


def source_paths(root, stem):
    match = STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"invalid DexYCB sample name: {stem}")
    subject, sequence, serial, frame = match.groups()
    camera_dir = root / subject / sequence / serial
    return (camera_dir / f"color_{frame}.jpg",
            camera_dir / f"aligned_depth_to_color_{frame}.png",
            root / "calibration/intrinsics" / f"{serial}_640x480.yml")


@lru_cache(maxsize=32)
def read_intrinsics(path):
    raw = path.read_bytes()
    # 官方 YAML 含 !!python/tuple；FullLoader 支持该标签而不使用 UnsafeLoader。
    color = yaml.full_load(raw)["color"]
    k = np.array([[color["fx"], 0, color["ppx"]],
                  [0, color["fy"], color["ppy"]], [0, 0, 1]], dtype=np.float64)
    return k, raw


def native_inputs(record, rgb, depth, k):
    """原始 RGBD 与已 canonicalize 的相机系 MANO 保持一致。"""
    if rgb is None or depth is None or rgb.dtype != np.uint8 or depth.dtype != np.uint16:
        raise ValueError("expected uint8 RGB and uint16 aligned depth")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth must have the same full resolution")
    side = record.get("source_mano_side")
    mode = record.get("canonicalization")
    if record.get("canonical_mano_side") != "right":
        raise ValueError("annotations must be in the canonical right-hand frame")
    if side == "left" and mode == "camera_x_reflection":
        return rgb[:, ::-1].copy(), depth[:, ::-1].copy(), reflect_intrinsics(k, rgb.shape[1])
    if side == "right" and mode == "none":
        return rgb.copy(), depth.copy(), k.copy()
    raise ValueError(f"unsupported side/canonicalization: {side}/{mode}")


def verify_old_camera(record, native_k, width, height):
    """由原转换脚本的裁剪/缩放关系交叉核对相机，避免串错序列或手侧。"""
    old = record["camera"]
    if old["width"] != 224 or old["height"] != 224:
        raise ValueError("annotation-dir must contain the 224x224 v2/v3 annotations")
    size = min(width, height)
    expected = native_k.copy()
    expected[0, 2] -= (width - size) // 2
    expected[1, 2] -= (height - size) // 2
    expected[:2] *= 224.0 / size
    if not np.allclose(expected, old["K"], atol=1e-4, rtol=1e-6):
        raise ValueError("native camera does not match annotation camera")


def png(array):
    ok, buf = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def mesh_mask(vertices, faces, k, width, height):
    """在完整矩形画布画三角面；画布外顶点不丢弃，由 OpenCV 裁到边界。"""
    projection = vertices @ k.T
    if not np.isfinite(projection).all() or np.any(projection[:, 2] <= 1e-6):
        raise ValueError("invalid mesh projection")
    uv = projection[:, :2] / projection[:, 2:]
    if np.max(np.abs(uv)) > 1e7:
        raise ValueError("mesh projection exceeds rasterizer bounds")
    triangles = np.round(uv).astype(np.int32)[np.asarray(faces, dtype=np.int64)]
    mask = np.zeros((height, width), dtype=np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 255)
    # 与旧 224 画布上的 5x5、两次膨胀近似保持相同原图覆盖范围。
    radius = max(1, round(2 * min(width, height) / 224))
    return cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8), iterations=2)


def signature():
    paths = [Path(__file__), REPO / "scripts/conversion_common.py",
             REPO / "scripts/repair_shape_gt_geometry.py", REPO / "src/models/mano.py",
             REPO / "src/utils/transform_utils.py"]
    paths += sorted((REPO / "assets/mano").rglob("*.pkl"))
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def read_frame(annotation_root, raw_root, output, rel, implementation):
    rgb_path, depth_path, k_path = source_paths(raw_root, rel.stem)
    paths = [annotation_root / rel, rgb_path, depth_path]
    for path, root in zip(paths, [annotation_root, raw_root, raw_root]):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"source path escapes root: {path}")
    annotation_bytes, rgb_bytes, depth_bytes = [p.read_bytes() for p in paths]
    k, k_bytes = read_intrinsics(k_path)
    marker = {"version": VERSION, "implementation_sha256": implementation,
              "source_sha256": [hashlib.sha256(b).hexdigest()
                                for b in [annotation_bytes, rgb_bytes, depth_bytes, k_bytes]],
              "shape_source": "grasp.shape_gt", "rgb_encoding": "png"}
    target = output / rel
    if target.exists():
        with target.open("rb") as f:
            old_output = pickle.load(f)
        if old_output.get("fullres_conversion") != marker:
            raise ValueError(f"existing output fingerprint mismatch: {target}; use a new output directory")
        return rel, None, marker
    record = pickle.loads(annotation_bytes)
    bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    rgb, depth, native_k = native_inputs(record, rgb, depth, k)
    height, width = depth.shape
    verify_old_camera(record, native_k, width, height)
    # 新图像坐标下的字段全部重建；其他参数和 3D 坐标系继承已验证标注。
    result = dict(record)
    result.pop("shape_gt_geometry_repair", None)
    result.pop("condition_point", None)
    result["camera"] = {"K": native_k, "width": width, "height": height}
    result["camera_original"] = {"K": native_k.copy(), "width": width, "height": height}
    result["image"] = png(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    result["depth"] = png(depth)
    result["schema_version"] = VERSION
    result["source_rgb_path"] = str(rgb_path.relative_to(raw_root))
    result["source_depth_path"] = str(depth_path.relative_to(raw_root))
    return rel, result, marker


def write_frame(item, output):
    rel, record, marker = item
    cam, grasp = record["camera"], record["grasp"]
    mask = mesh_mask(grasp["mesh_vertices"], grasp["mesh_faces"], cam["K"], cam["width"], cam["height"])
    record["object_mask"] = png(mask)
    record["fullres_conversion"] = marker
    atomic_write(output / rel, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-dir", type=Path, required=True, help="已修复的 v3 pkl 标注目录（也支持含 shape_gt 的 v2）")
    p.add_argument("--dexycb-root", type=Path, default=Path("/root/code/vepfs/dataset/dex-ycb"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--samples", type=Path, nargs="+", required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0=全量；正数=均匀抽样")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--io-workers", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=2)
    args = p.parse_args()
    if min(args.batch_size, args.io_workers, args.torch_threads) < 1 or args.max_samples < 0:
        p.error("invalid batch-size/io-workers/torch-threads/max-samples")
    annotation, raw, output = args.annotation_dir.resolve(), args.dexycb_root.resolve(), args.out_dir.resolve()
    for source in (annotation, raw):
        if not source.is_dir():
            p.error(f"source directory does not exist: {source}")
        if source == output or source in output.parents or output in source.parents:
            p.error("out-dir must be separate from both source directories")
    paths = collect_files(annotation, args.samples, args.max_samples)
    for rel in paths:
        source_paths(raw, rel.stem)
    implementation = signature()
    manifest = {"version": VERSION, "annotation_dir": str(annotation), "dexycb_root": str(raw),
                "implementation_sha256": implementation,
                "selection_sha256": hashlib.sha256("\n".join(x.as_posix() for x in paths).encode()).hexdigest(),
                "selected": len(paths)}
    manifest_path = output / "conversion_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("output manifest differs; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output directory is nonempty and has no conversion manifest")
    atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
    print(json.dumps({"selected": len(paths), "output_dir": str(output), "device": "cpu"}), flush=True)
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(1)
    mano = MANO().eval()
    written = resumed = 0
    start = time.monotonic()
    report = {**manifest, "output_dir": str(output), "status": "running"}
    try:
        with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
            for offset in range(0, len(paths), args.batch_size):
                jobs = list(pool.map(lambda rel: read_frame(annotation, raw, output, rel, implementation),
                                     paths[offset:offset + args.batch_size]))
                pending = [(rel, d, marker) for rel, d, marker in jobs if d is not None]
                resumed += len(jobs) - len(pending)
                if pending:
                    fixed = repair_batch([d for _, d, _ in pending], mano, "cpu")
                    items = [(rel, d, marker) for (rel, _, marker), d in zip(pending, fixed)]
                    list(pool.map(lambda item: write_frame(item, output), items))
                    written += len(items)
                if offset == 0 or (offset // args.batch_size) % 20 == 0 or written + resumed == len(paths):
                    print(json.dumps({"processed": written + resumed, "total": len(paths),
                                      "written": written, "resumed": resumed,
                                      "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
        atomic_write(output / "samples.txt", ("\n".join(x.with_suffix("").as_posix() for x in paths) + "\n").encode())
        report["status"] = "success"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report.update(written_this_run=written, resumed_this_run=resumed,
                      elapsed_seconds=round(time.monotonic() - start, 3))
        atomic_write(output / "conversion_report.json", json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""HO3D full-resolution, native-shape conversion and independent data checks.

Run from the repository root. See docs/HO3D_RECONVERSION_V2.md.
No detector, GT crop, synthetic eval beta, or GT condition point is written.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SCHEMA = "ho3d_native_geometry_v1"
ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TIPS = [744, 333, 444, 555, 672]
TIP_JOINTS = [4, 8, 12, 16, 20]
C = np.diag([1., -1., -1.]).astype(np.float32)
DEPTH_SCALE = 0.00012498664727900177
_LAYER = None


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_lines(path):
    values = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"empty/duplicate sample list: {path}")
    for s in values:
        if Path(s).is_absolute() or ".." in Path(s).parts:
            raise ValueError(f"unsafe sample name: {s}")
    return values


def write_lines(path, lines):
    Path(path).write_text("".join(s + "\n" for s in lines))


def layer():
    global _LAYER
    if _LAYER is None:
        import torch
        from src.models.mano import MANO
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        _LAYER = MANO().mano_layer.eval()
    return _LAYER


def project(xyz, K):
    p = np.asarray(xyz) @ np.asarray(K).T
    if np.any(p[:, 2] <= 0):
        raise ValueError("nonpositive joint camera depth")
    return (p[:, :2] / p[:, 2:]).astype(np.float32)


def finite(value, shape, dtype=np.float32):
    a = np.asarray(value, dtype=dtype)
    if a.size != int(np.prod(shape)) or not np.isfinite(a).all():
        raise ValueError(f"invalid numeric annotation, expected {shape}")
    return a.reshape(shape)


def train_geometry(meta):
    """Decode parameters independently of the old converter, using real beta."""
    import torch
    l = layer()
    aa = finite(meta["handPose"], (16, 3))
    beta = finite(meta["handBeta"], (1, 10))
    trans = finite(meta["handTrans"], (3,))
    official = finite(meta["handJoints3D"], (21, 3))[ORDER] @ C.T
    rotations = np.stack([cv2.Rodrigues(x.astype(np.float64))[0] for x in aa]).astype(np.float32)
    rotations[0] = C @ rotations[0]
    with torch.no_grad():
        bt = torch.from_numpy(beta)
        shaped = l.th_v_template + torch.einsum("vck,bk->bvc", l.th_shapedirs, bt)
        offset = (l.th_J_regressor @ shaped)[0, 0].numpy()
        template_offset = (l.th_J_regressor @ l.th_v_template)[0, 0].numpy()
        wrist = (trans + offset) @ C.T
        out = l.skinning_layer(torch.from_numpy(rotations[None]), bt)
        verts = out["verts"][0].numpy() + wrist
        joints = out["joints"][0].numpy() + wrist
        default_joints = joints.copy()
        joints[TIP_JOINTS] = verts[TIPS]
    error = np.linalg.norm(joints - official, axis=1) * 1000
    # Annotation and parameter geometry should agree far below 0.1 mm.
    if error.max() > 0.1:
        raise ValueError(f"MANO/official joint inconsistency: max={error.max():.6f} mm")
    pose_cam = aa.copy()
    pose_cam[0] = cv2.Rodrigues(rotations[0].astype(np.float64))[0].reshape(3)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3], transform[:3, 3] = rotations[0], wrist
    g = dict(t=wrist[None].astype(np.float32), R_6d=rotations[0, :, :2].reshape(1, 6),
             pose_6d=rotations[1:, :, :2].reshape(1, 15, 6),
             pose=pose_cam[1:].reshape(1, 15, 3), shape=beta.copy(), shape_gt=beta.copy(),
             T_camera_wrist=transform, landmarks_3d=official.astype(np.float32),
             mesh_vertices=verts.astype(np.float32), mesh_faces=l.th_faces.numpy().copy())
    audit = dict(joint_max_mm=float(error.max()), joint_mean_mm=float(error.mean()),
                 template_wrist_error_mm=float(np.linalg.norm((trans + template_offset) @ C.T - official[0]) * 1000),
                 default_tip_error_mm=(np.linalg.norm(default_joints[TIP_JOINTS] - official[TIP_JOINTS], axis=1) * 1000).tolist())
    return g, audit


def eval_arrays(root):
    # Convert immediately to compact arrays; do not retain millions of Python floats.
    with (root / "evaluation_xyz.json").open() as f:
        joints = np.asarray(json.load(f), dtype=np.float32)
    with (root / "evaluation_verts.json").open() as f:
        verts = np.asarray(json.load(f), dtype=np.float32)
    n = len(read_lines(root / "evaluation.txt"))
    if joints.shape != (n, 21, 3) or verts.shape != (n, 778, 3):
        raise ValueError("official evaluation list / joints / vertices length mismatch")
    return joints, verts


def convert_one(task):
    root_s, out_s, split, line, index, xyz, verts = task
    root, output = Path(root_s), Path(out_s)
    seq, frame = line.split("/")
    stem = f"ho3d_{'eval_' if split == 'evaluation' else ''}{seq}_{frame}"
    dest = output / f"{stem}.pkl"
    try:
        if dest.exists():
            with dest.open("rb") as f:
                saved = pickle.load(f)
            if (saved.get("schema_version") != SCHEMA or saved.get("source_sample") != line
                    or saved.get("official_index") != index or saved.get("source_split") != split):
                raise ValueError("existing output does not match conversion manifest")
            return stem, None, saved.get("conversion_audit", {})
        base = root / split / seq
        with (base / "meta" / f"{frame}.pkl").open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        image_bytes = (base / "rgb" / f"{frame}.jpg").read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        rawdepth = cv2.imread(str(base / "depth" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or rawdepth is None or (rawdepth.ndim != 3 or rawdepth.shape[:2] != image.shape[:2] or rawdepth.shape[2] not in (3, 4)):
            raise ValueError("missing/bad RGB or BGR/BGRA encoded depth")
        h, w = image.shape[:2]
        K = finite(meta["camMat"], (3, 3), dtype=np.float64)
        depth_m = (rawdepth[:, :, 1].astype(np.uint32) * 256 + rawdepth[:, :, 2]) * DEPTH_SCALE
        depth_mm = np.rint(np.clip(depth_m * 1000, 0, 65535)).astype(np.uint16)
        ok, buf = cv2.imencode(".png", depth_mm)
        if not ok:
            raise ValueError("depth encoding failed")
        record = dict(schema_version=SCHEMA, source_dataset="HO3D_v3", source_sample=line,
                      source_split=split, official_index=index, source_mano_side="right",
                      canonical_mano_side="right", canonicalization="none",
                      mano_parameter_convention="source_beta_canonical_pose_v1",
                      joint_convention="ho3d_official_v1", ho3d_tip_vertex_ids=TIPS,
                      object_name=f"ho3d_{seq}", frame_index=int(frame), grasp_index=0,
                      camera=dict(K=K, width=w, height=h), camera_original=dict(K=K.copy(), width=w, height=h),
                      image=image_bytes, depth=buf.tobytes(), object_mask=b"",
                      condition_point=None, grasp=None)
        audit = {}
        if split == "train":
            g, audit = train_geometry(meta)
            g["landmarks_2d"] = project(g["landmarks_3d"], K)
            record["grasp"] = g
            record["joint_order"] = "wrist_thumb_index_middle_ring_pinky"
        else:
            # Preserve official RAW order; the Dataset reorders exactly once.
            record["joints_gt"] = finite(xyz, (21, 3)) @ C.T
            record["verts_gt"] = finite(verts, (778, 3)) @ C.T
            record["joint_order"] = "ho3d_raw"
            if any(np.asarray(meta.get(k)).size > 1 for k in ("handPose", "handBeta", "handTrans")):
                raise ValueError("unexpected evaluation MANO labels: inspect protocol")
            tip_error = np.linalg.norm(record["joints_gt"][ORDER][TIP_JOINTS] - record["verts_gt"][TIPS], axis=1) * 1000
            if tip_error.max() > 0.1:
                raise ValueError(f"evaluation joint/vertex convention mismatch: {tip_error.max()} mm")
            audit = dict(tip_max_mm=float(tip_error.max()))
        record["conversion_audit"] = audit
        tmp = dest.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dest)
        return stem, None, audit
    except Exception as e:
        return stem, f"{type(e).__name__}: {e}", {}


def convert(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".convert.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lines = read_lines(args.raw_root / f"{args.split}.txt")
    ids = np.arange(len(lines))
    if args.limit:
        ids = np.linspace(0, len(lines) - 1, min(args.limit, len(lines)), dtype=int)
    asset = Path("assets/mano/models/MANO_RIGHT.pkl")
    identity = dict(schema_version=SCHEMA, raw_root=str(args.raw_root.resolve()),
                    split=args.split, official_count=len(lines), selected_count=len(ids),
                    list_sha256=sha(args.raw_root / f"{args.split}.txt"),
                    script_sha256=sha(__file__), mano_sha256=sha(asset), limit=args.limit)
    if args.split == "evaluation":
        identity["evaluation_xyz_sha256"] = sha(args.raw_root / "evaluation_xyz.json")
        identity["evaluation_verts_sha256"] = sha(args.raw_root / "evaluation_verts.json")
    identity_file = args.output / "conversion_identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError("output identity changed; use a NEW output directory")
    elif any(args.output.glob("*.pkl")):
        raise ValueError("refusing to mix with an existing dataset without a manifest")
    write_json(identity_file, identity)
    write_json(args.output / "conversion_report.json", dict(status="running", **identity))
    xyz, verts = eval_arrays(args.raw_root) if args.split == "evaluation" else (None, None)
    tasks = ((str(args.raw_root), str(args.output), args.split, lines[i], int(i),
              None if xyz is None else xyz[i], None if verts is None else verts[i]) for i in ids)
    good, errors, audits = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (stem, err, audit) in enumerate(pool.map(convert_one, tasks, chunksize=32), 1):
            if err:
                errors.append(dict(sample=stem, error=err))
                print(f"ERROR {stem}: {err}", flush=True)
            else:
                good.append(stem)
                audits.append(audit)
            if n % 1000 == 0:
                print(f"{n}/{len(ids)}, failures={len(errors)}", flush=True)
    write_lines(args.output / "samples.txt", good)
    report = dict(status="failed" if errors else "complete", **identity,
                  converted=len(good), failed=len(errors), errors=errors)
    if audits:
        report["max_geometry_error_mm"] = max(a.get("joint_max_mm", a.get("tip_max_mm", 0)) for a in audits)
        if args.split == "train":
            report["old_template_wrist_mean_error_mm"] = float(np.mean([a["template_wrist_error_mm"] for a in audits]))
    write_json(args.output / "conversion_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
    if errors:
        raise SystemExit("Conversion incomplete: review errors; no failed sample was silently accepted.")


def load_record(root, stem):
    with (root / f"{stem}.pkl").open("rb") as f:
        return pickle.load(f)


def selected(stems, n, seed):
    ids = np.random.default_rng(seed).choice(len(stems), min(n, len(stems)), replace=False)
    return [stems[i] for i in ids]


def draw_joints(img, xy, color):
    xy = np.asarray(xy)
    for start in [1, 5, 9, 13, 17]:
        chain = [0] + list(range(start, start + 4))
        for a, b in zip(chain, chain[1:]):
            if np.isfinite(xy[[a, b]]).all() and np.abs(xy[[a, b]]).max() < 100000:
                cv2.line(img, tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int)), color, 1)
    for p in xy:
        if np.isfinite(p).all() and np.abs(p).max() < 100000:
            cv2.circle(img, tuple(np.rint(p).astype(int)), 2, color, -1)


def check(args):
    report = json.loads((args.dataset_root / "conversion_report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("conversion is incomplete")
    stems = read_lines(args.dataset_root / "samples.txt")
    if len(stems) != report["selected_count"]:
        raise ValueError("sample count mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    eval_xyz, eval_verts = eval_arrays(args.raw_root) if report["split"] == "evaluation" else (None, None)
    results = []
    for n, stem in enumerate(selected(stems, args.samples, args.seed)):
        d = load_record(args.dataset_root, stem)
        seq, frame = d["source_sample"].split("/")
        base = args.raw_root / d["source_split"] / seq
        meta = pickle.load(open(base / "meta" / f"{frame}.pkl", "rb"), encoding="latin1")
        assert d["image"] == (base / "rgb" / f"{frame}.jpg").read_bytes(), stem
        image = cv2.imdecode(np.frombuffer(d["image"], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(d["depth"], np.uint8), cv2.IMREAD_UNCHANGED)
        K = d["camera"]["K"]
        assert np.allclose(K, meta["camMat"], atol=1e-5, rtol=0), stem
        assert image.shape[:2] == depth.shape == (d["camera"]["height"], d["camera"]["width"])
        assert depth.dtype == np.uint16 and d["condition_point"] is None and not d["object_mask"]
        raw = cv2.imread(str(base / "depth" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        expected_depth = np.rint(np.clip((raw[:,:,1].astype(np.uint32)*256 + raw[:,:,2])*DEPTH_SCALE*1000,0,65535)).astype(np.uint16)
        assert np.array_equal(depth, expected_depth)
        if d["grasp"] is not None:
            g, audit = train_geometry(meta)
            for key in ("t", "R_6d", "pose_6d", "shape", "shape_gt", "mesh_vertices", "landmarks_3d"):
                assert np.allclose(g[key], d["grasp"][key], atol=1e-6, rtol=0), (stem,key)
            assert np.allclose(d["grasp"]["landmarks_2d"], project(g["landmarks_3d"], K), atol=1e-4)
            xyz = g["landmarks_3d"]
        else:
            i = d["official_index"]
            assert read_lines(args.raw_root / "evaluation.txt")[i] == d["source_sample"]
            assert np.array_equal(d["joints_gt"], eval_xyz[i] @ C.T)
            assert np.array_equal(d["verts_gt"], eval_verts[i] @ C.T)
            xyz = d["joints_gt"][ORDER]
            audit = d["conversion_audit"]
        results.append(dict(sample=stem, **audit))
        if n < args.visualize:
            draw_joints(image, project(xyz,K), (0,220,0))
            cv2.putText(image, stem, (8,22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),1)
            cv2.imwrite(str(args.output / f"{n:02d}_{stem}.jpg"), image)
    write_json(args.output / "check_report.json", dict(status="passed", checked=len(results), seed=args.seed, samples=results))
    (args.output / "README.md").write_text("绿色骨架：官方 HO3D 3D 关节点投影，仅用于核验标签，不作为部署输入。图片是原始分辨率，没有预先裁剪或数据增强。\n")
    print(f"PASS: {len(results)} randomly sampled records; visualizations: {args.output}")


def splits(args):
    train = read_lines(args.train_root / "samples.txt")
    evaluation = read_lines(args.eval_root / "samples.txt")
    for root in (args.train_root, args.eval_root):
        r = json.loads((root / "conversion_report.json").read_text())
        if r["status"] != "complete" or r["limit"] is not None or r["converted"] != r["official_count"]:
            raise ValueError("split generation requires COMPLETE, full official conversions")
    def sequence(s):
        return s.removeprefix("ho3d_").rsplit("_",1)[0]
    val_sequences = {sequence(s) for s in read_lines(args.old_val)}
    if not val_sequences.issubset({sequence(s) for s in train}):
        raise ValueError("old held-out sequence missing from new train pool")
    val = [s for s in train if sequence(s) in val_sequences]
    tr = [s for s in train if sequence(s) not in val_sequences]
    assert tr and val and not set(tr) & set(val)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("train",tr),("val",val),("eval",evaluation),("trainval",train)):
        dest = args.output / f"ho3d_{name}.txt"
        if dest.exists() and read_lines(dest) != values:
            raise ValueError(f"refusing to overwrite a different split: {dest}")
        write_lines(dest, values)
    write_json(args.output / "split_report.json", dict(train=len(tr),val=len(val),evaluation=len(evaluation),
               heldout_sequences=sorted(val_sequences),old_val=str(args.old_val), old_val_sha256=sha(args.old_val),
               policy="preserve old val sequence identities, recover every official frame"))
    print(f"train={len(tr)}, val={len(val)}, evaluation={len(evaluation)}; heldout={sorted(val_sequences)}")


def stats(args):
    stems = read_lines(args.samples_file)
    n, mean, m2 = 0, np.zeros(109), np.zeros(109)
    for stem in stems:
        d = load_record(args.dataset_root, stem)
        if d["source_split"] != "train" or d["grasp"] is None:
            raise ValueError("normalization requires annotated train-only samples")
        g = d["grasp"]
        x = np.concatenate([g[k].ravel() for k in ("t","R_6d","pose_6d","shape_gt")]).astype(np.float64)
        assert x.shape == (109,) and np.isfinite(x).all()
        n += 1
        delta = x-mean
        mean += delta/n
        m2 += delta*(x-mean)
    std = np.sqrt(m2/n)
    std = np.where(std > 1e-6, std, 1.)
    result = {k:dict(mean=mean[a:b].tolist(),std=std[a:b].tolist()) for k,a,b in
              [("translation",0,3),("wrist_rot",3,9),("finger_rot",9,99),("shape",99,109)]}
    write_json(args.output,result)
    write_json(args.output.with_suffix(".audit.json"),dict(n=n,dataset_root=str(args.dataset_root),
              samples_file=str(args.samples_file),samples_sha256=sha(args.samples_file),mean=mean.tolist(),m2=m2.tolist()))
    print(f"PASS: strict 109D statistics from {n} samples -> {args.output}")


def cache_check(args):
    from src.cache_train_conditions import validate
    stems = read_lines(args.samples_file)
    meta = json.loads((args.cache.parent / "metadata.json").read_text())
    assert Path(meta["dataset_root"]).resolve() == args.dataset_root.resolve()
    assert meta["samples_sha256"] == sha(args.samples_file)
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    validate(data,stems,0)
    args.output.mkdir(parents=True,exist_ok=True)
    lookup = {s:i for i,s in enumerate(stems)}
    for n, stem in enumerate(selected(stems,args.samples,args.seed)):
        i=lookup[stem]; d=load_record(args.dataset_root,stem)
        image=cv2.imdecode(np.frombuffer(d["image"],np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2]
        assert np.array_equal(data["image_size_wh"][i],[w,h])
        if n < args.visualize:
            if data["detector_hit"][i]:
                for key,color in (("detector_bbox_xyxy",(0,0,255)),("crop_bbox_xyxy",(0,220,0))):
                    b=np.rint(data[key][i]).astype(int)
                    cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),color,2)
                xy=data["keypoints_xy"][i].copy()
                xy[data["keypoint_scores"][i]<.1]=np.nan
                draw_joints(image,xy,(255,0,255))
            cv2.putText(image,f"{stem} hit={bool(data['detector_hit'][i])}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            cv2.imwrite(str(args.output/f"{n:02d}_{stem}.jpg"),image)
    write_json(args.output/"cache_check_report.json",dict(status="passed",n_samples=len(stems),
               detector_hits=int(data["detector_hit"].sum()),detector_misses=int((~data["detector_hit"]).sum()),
               checked_images=min(args.samples,len(stems)),cache_sha256=sha(args.cache)))
    (args.output/"README.md").write_text("红框：WiLoR/YOLO 原始检测框。绿框：扩大 1.5 倍后的正方形 RGB crop，可超出原图并补边。紫色：RTMPose 预测骨架，只绘制置信度 ≥0.1 的点。hit=False：检测失败，保留样本并回退全图，不使用 GT 补框。所有坐标均在原图空间；此图检查原始缓存，不是增强后的图。\n")
    print(f"PASS cache: {len(stems)} rows, {int((~data['detector_hit']).sum())} misses retained")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("convert"); c.set_defaults(func=convert)
    c.add_argument("--raw-root",type=Path,required=True); c.add_argument("--output",type=Path,required=True)
    c.add_argument("--split",choices=["train","evaluation"],required=True)
    c.add_argument("--workers",type=int,default=4); c.add_argument("--limit",type=int)
    c=sub.add_parser("check"); c.set_defaults(func=check)
    c.add_argument("--raw-root",type=Path,required=True)
    c=sub.add_parser("splits"); c.set_defaults(func=splits)
    for arg in ("train-root","eval-root","old-val","output"): c.add_argument("--"+arg,type=Path,required=True)
    c=sub.add_parser("stats"); c.set_defaults(func=stats)
    c=sub.add_parser("cache-check"); c.set_defaults(func=cache_check)
    c.add_argument("--cache",type=Path,required=True)
    for name in ("check","stats","cache-check"):
        c=sub.choices[name]
        c.add_argument("--dataset-root",type=Path,required=True); c.add_argument("--output",type=Path,required=True)
        if name!="check": c.add_argument("--samples-file",type=Path,required=True)
        if name!="stats":
            c.add_argument("--samples",type=int,default=256); c.add_argument("--visualize",type=int,default=20)
            c.add_argument("--seed",type=int,default=20260919)
    args=p.parse_args()
    if getattr(args,"limit",None) is not None and args.limit < 1: p.error("--limit must be positive")
    if getattr(args,"workers",1)<1: p.error("--workers must be positive")
    args.func(args)


if __name__=="__main__":
    main()

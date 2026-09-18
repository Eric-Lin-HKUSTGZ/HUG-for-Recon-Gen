"""Compare MediaPipe Hands and RTMPose-m Hand5 on converted DexYCB.

The benchmark is split into three commands because the project keeps MediaPipe
and MMPose in separate conda environments:

1. ``prepare`` selects samples, runs the existing YOLO hand crop and MediaPipe.
2. ``rtmpose`` runs RTMPose-m Hand5 on exactly the same crop boxes.
3. ``report`` evaluates both predictions against DexYCB 2D GT and renders plots.

Missing predictions count as incorrect in the full-dataset PCK/AUC metrics.
Conditional EPE/NME is also reported so coverage and localization accuracy are
not conflated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/root/code/vepfs/dataset/hand_recon_hug/dexycb_v4_fullres_shape_gt"
)
DEFAULT_SAMPLES_FILE = Path(
    "/root/code/vepfs/dataset/hand_recon_hug/splits_v2/dexycb_test.clean.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/code/vepfs/HUG-for-Recon-Gen/hand_keypoint_benchmark/"
    "dexycb_test_detector_1000"
)
DEFAULT_DETECTOR = Path("/root/code/vepfs/GPGFormer/weights/detector.pt")
DEFAULT_RTMPOSE_CONFIG = Path(
    "/root/code/vepfs/third_party/mmpose-1.3.2/configs/"
    "hand_2d_keypoint/rtmpose/hand5/"
    "rtmpose-m_8xb256-210e_hand5-256x256.py"
)
DEFAULT_RTMPOSE_CHECKPOINT = Path(
    "/root/code/vepfs/HUG-for-Recon-Gen/rtmpose/"
    "rtmpose-m-hand5-256x256.pth"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
JOINT_NAMES = [
    "wrist",
    "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4",
]
METHOD_COLORS = {
    "gt": (40, 220, 40),
    "mediapipe": (30, 190, 255),
    "rtmpose": (230, 80, 230),
}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_stems(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def select_stems(
    stems: Sequence[str], max_samples: int, sampling: str, seed: int
) -> list[str]:
    if max_samples <= 0 or max_samples >= len(stems):
        return list(stems)
    if sampling == "head":
        return list(stems[:max_samples])
    if sampling == "random":
        indices = sorted(random.Random(seed).sample(range(len(stems)), max_samples))
    else:
        indices = np.linspace(0, len(stems) - 1, max_samples, dtype=np.int64).tolist()
    return [stems[int(index)] for index in indices]


def load_sample(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    image = cv2.imdecode(
        np.frombuffer(sample["image"], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"failed to decode RGB from {path}")
    return sample, image


def tight_bbox(points: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    keep = np.isfinite(xy).all(axis=1)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool).reshape(-1)
    if keep.sum() < 2:
        raise ValueError("at least two valid points are required for a bbox")
    xy = xy[keep]
    return np.array(
        [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()],
        dtype=np.float32,
    )


def expanded_square_bbox(
    bbox: np.ndarray, expand: float = 1.5, min_side: float = 24.0
) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(float(x2 - x1), float(y2 - y1), float(min_side)) * expand
    return np.array(
        [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2],
        dtype=np.float32,
    )


def crop_affine(bbox: np.ndarray, output_size: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4)
    sx = float(output_size - 1) / max(float(x2 - x1), 1.0)
    sy = float(output_size - 1) / max(float(y2 - y1), 1.0)
    return np.array([[sx, 0.0, -sx * x1], [0.0, sy, -sy * y1]], np.float32)


def transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    xy1 = np.concatenate([xy, np.ones((len(xy), 1), np.float32)], axis=1)
    return (xy1 @ np.asarray(affine, np.float32).T).astype(np.float32)


def invert_affine(affine: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = np.asarray(affine, dtype=np.float32)
    return np.linalg.inv(matrix).astype(np.float32)[:2]


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(a, np.float32).reshape(4)
    bx1, by1, bx2, by2 = np.asarray(b, np.float32).reshape(4)
    iw = max(0.0, min(float(ax2), float(bx2)) - max(float(ax1), float(bx1)))
    ih = max(0.0, min(float(ay2), float(by2)) - max(float(ay1), float(by1)))
    inter = iw * ih
    aa = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    ab = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    return inter / max(aa + ab - inter, 1e-9)


def choose_detector_bbox(result: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None, {"returned": False}
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
    right = np.flatnonzero(classes > 0.5)
    candidates = right if right.size else np.arange(len(scores))
    index = int(candidates[np.argmax(scores[candidates])])
    return boxes[index], {
        "returned": True,
        "score": float(scores[index]),
        "class_id": int(round(float(classes[index]))),
        "num_boxes": int(len(boxes)),
    }


def run_mediapipe(
    hands: Any, bgr: np.ndarray, bbox: np.ndarray, input_size: int
) -> dict[str, Any]:
    affine = crop_affine(bbox, input_size)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crop = cv2.warpAffine(
        rgb,
        affine,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    started = time.perf_counter()
    result = hands.process(np.ascontiguousarray(crop))
    latency_ms = 1000.0 * (time.perf_counter() - started)
    if not result.multi_hand_landmarks:
        return {"returned": False, "latency_ms": latency_ms}
    landmarks = result.multi_hand_landmarks[0].landmark
    crop_xy = np.asarray(
        [[lm.x * input_size, lm.y * input_size] for lm in landmarks],
        dtype=np.float32,
    )
    pred_xy = transform_points(crop_xy, invert_affine(affine))
    handedness_score = None
    if result.multi_handedness and result.multi_handedness[0].classification:
        handedness_score = float(result.multi_handedness[0].classification[0].score)
    return {
        "returned": True,
        "keypoints_xy": pred_xy.tolist(),
        "latency_ms": latency_ms,
        "handedness_score": handedness_score,
    }


def command_prepare(args: argparse.Namespace) -> None:
    import mediapipe as mp

    all_stems = read_stems(args.samples_file)
    stems = select_stems(all_stems, args.max_samples, args.sampling, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    detector = None
    if args.bbox_source == "detector":
        from ultralytics import YOLO

        if not args.detector_weights.is_file():
            raise FileNotFoundError(args.detector_weights)
        detector = YOLO(str(args.detector_weights))

    manifests: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    detector_latencies: list[float] = []
    started_all = time.perf_counter()

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.mediapipe_min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as hands:
        for index, stem in enumerate(stems):
            pkl_path = args.dataset_root / f"{stem}.pkl"
            record: dict[str, Any] = {
                "index": index,
                "sample": stem,
                "pkl_path": str(pkl_path),
            }
            prediction: dict[str, Any] = {"sample": stem, "returned": False}
            try:
                sample, bgr = load_sample(pkl_path)
                height, width = bgr.shape[:2]
                gt = np.asarray(sample["grasp"]["landmarks_2d"], np.float32)
                if gt.shape != (21, 2):
                    raise ValueError(f"expected GT shape (21, 2), got {gt.shape}")
                gt_finite = np.isfinite(gt).all(axis=1)
                gt_visible = (
                    gt_finite
                    & (gt[:, 0] >= 0)
                    & (gt[:, 0] < width)
                    & (gt[:, 1] >= 0)
                    & (gt[:, 1] < height)
                )
                gt_box = tight_bbox(gt, gt_finite)

                detector_info: dict[str, Any]
                raw_bbox: np.ndarray | None
                if detector is None:
                    raw_bbox = gt_box
                    detector_info = {"returned": True, "source": "gt"}
                else:
                    det_started = time.perf_counter()
                    det_result = detector(
                        bgr,
                        conf=args.detector_conf,
                        iou=args.detector_iou,
                        device=args.detector_device,
                        verbose=False,
                    )[0]
                    detector_latency = 1000.0 * (time.perf_counter() - det_started)
                    detector_latencies.append(detector_latency)
                    raw_bbox, detector_info = choose_detector_bbox(det_result)
                    detector_info["latency_ms"] = detector_latency
                    detector_info["source"] = "detector"

                if raw_bbox is None:
                    crop_bbox = np.array(
                        [0.0, 0.0, float(width - 1), float(height - 1)], np.float32
                    )
                    crop_source = "full_fallback"
                else:
                    crop_bbox = expanded_square_bbox(raw_bbox, args.bbox_expand)
                    crop_source = args.bbox_source
                    detector_info["bbox_xyxy"] = raw_bbox.tolist()
                    detector_info["bbox_iou_gt"] = bbox_iou(raw_bbox, gt_box)

                record.update(
                    {
                        "image_size_wh": [width, height],
                        "gt_keypoints_xy": gt.tolist(),
                        "gt_valid": gt_visible.tolist(),
                        "gt_bbox_xyxy": gt_box.tolist(),
                        "crop_bbox_xyxy": crop_bbox.tolist(),
                        "crop_source": crop_source,
                        "detector": detector_info,
                    }
                )
                prediction.update(
                    run_mediapipe(hands, bgr, crop_bbox, args.mediapipe_input_size)
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                record["error"] = message
                prediction["error"] = message
            manifests.append(record)
            predictions.append(prediction)
            if (index + 1) % args.progress_every == 0 or index + 1 == len(stems):
                returned = sum(bool(row.get("returned")) for row in predictions)
                print(
                    f"[prepare] {index + 1}/{len(stems)} "
                    f"mediapipe={returned/(index+1):.3f}",
                    flush=True,
                )

    jsonl_write(args.output_root / "manifest.jsonl", manifests)
    jsonl_write(args.output_root / "mediapipe_predictions.jsonl", predictions)
    valid_manifest = [row for row in manifests if "error" not in row]
    detector_returned = sum(
        bool(row.get("detector", {}).get("returned")) for row in valid_manifest
    )
    prepare_summary = {
        "dataset_root": str(args.dataset_root),
        "samples_file": str(args.samples_file),
        "available_samples": len(all_stems),
        "selected_samples": len(stems),
        "valid_samples": len(valid_manifest),
        "sampling": args.sampling,
        "seed": args.seed,
        "bbox_source": args.bbox_source,
        "bbox_expand": args.bbox_expand,
        "detector_weights": str(args.detector_weights),
        "detector_return_rate": (
            detector_returned / len(valid_manifest) if valid_manifest else 0.0
        ),
        "detector_latency_ms_mean": (
            float(np.mean(detector_latencies)) if detector_latencies else None
        ),
        "mediapipe_version": mp.__version__,
        "mediapipe_input_size": args.mediapipe_input_size,
        "mediapipe_return_rate": (
            sum(bool(row.get("returned")) for row in predictions) / len(predictions)
            if predictions else 0.0
        ),
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    json_dump(args.output_root / "prepare_summary.json", prepare_summary)
    print(json.dumps(prepare_summary, indent=2), flush=True)


def _rtmpose_batch(
    model: Any, pipeline: Any, pseudo_collate: Any, items: Sequence[tuple[dict, np.ndarray]]
) -> list[Any]:
    data_list = []
    for manifest, bgr in items:
        bbox = np.asarray(manifest["crop_bbox_xyxy"], np.float32)[None]
        data_info: dict[str, Any] = {
            "img": bgr,
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
        }
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))
    return model.test_step(pseudo_collate(data_list))


def command_rtmpose(args: argparse.Namespace) -> None:
    # OpenMMLab checkpoints contain trusted numpy metadata. PyTorch >=2.6
    # otherwise changes torch.load to weights_only=True inside old MMEngine.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmpose.apis import init_model

    if not args.rtmpose_config.is_file():
        raise FileNotFoundError(args.rtmpose_config)
    if not args.rtmpose_checkpoint.is_file():
        raise FileNotFoundError(args.rtmpose_checkpoint)

    manifests = jsonl_read(args.output_root / "manifest.jsonl")
    model = init_model(
        str(args.rtmpose_config),
        str(args.rtmpose_checkpoint),
        device=args.device,
        # The Hand5 config points to MMDetection's identical CSPNeXt. MMPose's
        # local implementation avoids requiring unused compiled MMCV ops.
        cfg_options={"model.backbone._scope_": "mmpose"},
    )
    init_default_scope(model.cfg.get("default_scope", "mmpose"))
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    predictions: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    warmed_up = False
    for start in range(0, len(manifests), args.batch_size):
        chunk = manifests[start : start + args.batch_size]
        valid_items: list[tuple[dict, np.ndarray]] = []
        valid_indices: list[int] = []
        chunk_predictions = [
            {"sample": row["sample"], "returned": False} for row in chunk
        ]
        for local_index, manifest in enumerate(chunk):
            if "error" in manifest:
                chunk_predictions[local_index]["error"] = manifest["error"]
                continue
            try:
                _, bgr = load_sample(Path(manifest["pkl_path"]))
                valid_items.append((manifest, bgr))
                valid_indices.append(local_index)
            except Exception as error:
                chunk_predictions[local_index]["error"] = (
                    f"{type(error).__name__}: {error}"
                )
        if valid_items:
            if not warmed_up:
                _rtmpose_batch(model, pipeline, pseudo_collate, valid_items[:1])
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                warmed_up = True
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            results = _rtmpose_batch(model, pipeline, pseudo_collate, valid_items)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_per_sample = 1000.0 * (time.perf_counter() - started) / len(results)
            for local_index, result in zip(valid_indices, results):
                instance = result.pred_instances
                keypoints = np.asarray(instance.keypoints[0], np.float32)
                scores = np.asarray(instance.keypoint_scores[0], np.float32)
                chunk_predictions[local_index].update(
                    {
                        "returned": keypoints.shape == (21, 2)
                        and bool(np.isfinite(keypoints).all()),
                        "keypoints_xy": keypoints.tolist(),
                        "keypoint_scores": scores.tolist(),
                        "latency_ms": latency_per_sample,
                    }
                )
        predictions.extend(chunk_predictions)
        returned = sum(bool(row.get("returned")) for row in predictions)
        print(
            f"[rtmpose] {len(predictions)}/{len(manifests)} "
            f"returned={returned/len(predictions):.3f}",
            flush=True,
        )

    jsonl_write(args.output_root / "rtmpose_predictions.jsonl", predictions)
    latencies = [
        float(row["latency_ms"]) for row in predictions if row.get("latency_ms") is not None
    ]
    summary = {
        "rtmpose_config": str(args.rtmpose_config),
        "rtmpose_checkpoint": str(args.rtmpose_checkpoint),
        "mmpose_version": __import__("mmpose").__version__,
        "torch_version": torch.__version__,
        "device": args.device,
        "batch_size": args.batch_size,
        "samples": len(predictions),
        "return_rate": (
            sum(bool(row.get("returned")) for row in predictions) / len(predictions)
            if predictions else 0.0
        ),
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_median": float(np.median(latencies)) if latencies else None,
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    json_dump(args.output_root / "rtmpose_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def sample_errors(
    manifest: dict[str, Any], prediction: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray] | None:
    if not prediction.get("returned"):
        return None
    gt = np.asarray(manifest["gt_keypoints_xy"], np.float32)
    pred = np.asarray(prediction.get("keypoints_xy"), np.float32)
    valid = np.asarray(manifest["gt_valid"], dtype=bool)
    if pred.shape != (21, 2) or valid.sum() < 2 or not np.isfinite(pred).all():
        return None
    # Normalize with the full projected hand extent. Using only the in-frame
    # subset becomes unstable when a hand is almost outside the image and the
    # two remaining visible points are only a fraction of a pixel apart.
    if manifest.get("gt_bbox_xyxy") is not None:
        scale_box = np.asarray(manifest["gt_bbox_xyxy"], np.float32)
    else:
        scale_box = tight_bbox(gt, np.isfinite(gt).all(axis=1))
    scale = max(float(scale_box[2] - scale_box[0]), float(scale_box[3] - scale_box[1]), 1.0)
    errors_px = np.linalg.norm(pred[valid] - gt[valid], axis=1)
    return errors_px, errors_px / scale


def evaluate_method(
    manifests: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    by_sample = {row["sample"]: row for row in predictions}
    thresholds = np.linspace(0.0, 0.5, 101, dtype=np.float64)
    all_correct = np.zeros_like(thresholds)
    conditional_correct = np.zeros_like(thresholds)
    total_visible = 0
    conditional_visible = 0
    returned_samples = 0
    errors_px_all: list[np.ndarray] = []
    errors_n_all: list[np.ndarray] = []
    sample_nmes: list[float] = []
    per_joint_errors: list[list[float]] = [[] for _ in range(21)]
    latencies: list[float] = []
    evaluable_samples = 0

    for manifest in manifests:
        if "error" in manifest:
            continue
        valid = np.asarray(manifest["gt_valid"], dtype=bool)
        if valid.sum() < 2:
            continue
        evaluable_samples += 1
        total_visible += int(valid.sum())
        pred = by_sample.get(manifest["sample"], {"returned": False})
        if pred.get("latency_ms") is not None:
            latencies.append(float(pred["latency_ms"]))
        result = sample_errors(manifest, pred)
        if result is None:
            continue
        returned_samples += 1
        errors_px, errors_n = result
        conditional_visible += len(errors_n)
        errors_px_all.append(errors_px)
        errors_n_all.append(errors_n)
        sample_nmes.append(float(errors_n.mean()))
        conditional_correct += (errors_n[:, None] <= thresholds[None]).sum(axis=0)
        all_correct += (errors_n[:, None] <= thresholds[None]).sum(axis=0)
        pred_xy = np.asarray(pred["keypoints_xy"], np.float32)
        gt_xy = np.asarray(manifest["gt_keypoints_xy"], np.float32)
        for joint in np.flatnonzero(valid):
            per_joint_errors[int(joint)].append(
                float(np.linalg.norm(pred_xy[joint] - gt_xy[joint]))
            )

    valid_samples = evaluable_samples
    pck_all = all_correct / max(total_visible, 1)
    pck_cond = conditional_correct / max(conditional_visible, 1)
    flat_px = np.concatenate(errors_px_all) if errors_px_all else np.array([])
    flat_n = np.concatenate(errors_n_all) if errors_n_all else np.array([])

    def at(curve: np.ndarray, threshold: float) -> float:
        return float(curve[int(round(threshold / 0.005))])

    metrics = {
        "valid_samples": valid_samples,
        "returned_samples": returned_samples,
        "return_rate": returned_samples / max(valid_samples, 1),
        "epe_px_mean_conditional": float(flat_px.mean()) if len(flat_px) else None,
        "epe_px_median_conditional": float(np.median(flat_px)) if len(flat_px) else None,
        "nme_mean_conditional": float(flat_n.mean()) if len(flat_n) else None,
        "nme_median_per_sample_conditional": (
            float(np.median(sample_nmes)) if sample_nmes else None
        ),
        "nme_per_sample_percentiles_conditional": (
            {
                str(percentile): float(np.percentile(sample_nmes, percentile))
                for percentile in (50, 75, 90, 95, 99, 100)
            }
            if sample_nmes else {}
        ),
        "catastrophic_sample_rate_nme_gt_0.5": (
            sum(value > 0.5 for value in sample_nmes) / max(valid_samples, 1)
        ),
        "usable_sample_rate_nme_le_0_10": (
            sum(value <= 0.10 for value in sample_nmes) / max(valid_samples, 1)
        ),
        "pck_0.05_all": at(pck_all, 0.05),
        "pck_0.10_all": at(pck_all, 0.10),
        "pck_0.20_all": at(pck_all, 0.20),
        "pck_0.20_conditional": at(pck_cond, 0.20),
        "auc_0.5_all": float(np.trapz(pck_all, thresholds) / 0.5),
        "auc_0.5_conditional": float(np.trapz(pck_cond, thresholds) / 0.5),
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_median": float(np.median(latencies)) if latencies else None,
        "per_joint_epe_px_conditional": [
            float(np.mean(values)) if values else None for values in per_joint_errors
        ],
    }
    return metrics, thresholds, pck_all


def draw_skeleton(
    image: np.ndarray,
    points: np.ndarray | None,
    valid: np.ndarray | None,
    color: tuple[int, int, int],
) -> None:
    if points is None:
        return
    xy = np.asarray(points, np.float32).reshape(-1, 2)
    keep = np.isfinite(xy).all(axis=1)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    for a, b in HAND_CONNECTIONS:
        if keep[a] and keep[b]:
            cv2.line(image, tuple(np.round(xy[a]).astype(int)), tuple(np.round(xy[b]).astype(int)), color, 2, cv2.LINE_AA)
    for index, point in enumerate(xy):
        if keep[index]:
            cv2.circle(image, tuple(np.round(point).astype(int)), 3, color, -1, cv2.LINE_AA)


def text_banner(image: np.ndarray, title: str, lines: Sequence[str]) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, " | ".join(lines), (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def render_sample_comparison(
    manifest: dict[str, Any], methods: dict[str, dict[str, Any]]
) -> np.ndarray:
    _, bgr = load_sample(Path(manifest["pkl_path"]))
    gt = np.asarray(manifest["gt_keypoints_xy"], np.float32)
    valid = np.asarray(manifest["gt_valid"], dtype=bool)
    crop_box = np.asarray(manifest["crop_bbox_xyxy"], np.float32)
    panels = []
    for name in ("gt", "mediapipe", "rtmpose"):
        panel = bgr.copy()
        cv2.rectangle(
            panel,
            tuple(np.round(crop_box[:2]).astype(int)),
            tuple(np.round(crop_box[2:]).astype(int)),
            (0, 220, 220),
            2,
        )
        if name == "gt":
            points = gt
            pred_valid = valid
            lines = [f"visible={int(valid.sum())}", f"crop={manifest['crop_source']}"]
        else:
            pred = methods[name]
            points = (
                np.asarray(pred["keypoints_xy"], np.float32)
                if pred.get("returned")
                else None
            )
            pred_valid = np.ones(21, dtype=bool) if points is not None else None
            result = sample_errors(manifest, pred)
            if result is None:
                lines = ["MISS"]
            else:
                errors_px, errors_n = result
                lines = [f"EPE={errors_px.mean():.1f}px", f"NME={errors_n.mean():.3f}"]
        draw_skeleton(panel, points, pred_valid, METHOD_COLORS[name])
        panel = text_banner(panel, name.upper(), lines)
        panels.append(cv2.resize(panel, (400, 300), interpolation=cv2.INTER_AREA))
    return np.concatenate(panels, axis=1)


def make_contact_sheet(images: Sequence[np.ndarray], output: Path, columns: int = 2) -> None:
    if not images:
        return
    height, width = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    sheet = np.full((rows * height, columns * width, 3), 245, np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def select_visual_samples(
    manifests: Sequence[dict[str, Any]], methods: dict[str, dict[str, dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    scored = []
    for manifest in manifests:
        if "error" in manifest:
            continue
        mp_result = sample_errors(manifest, methods["mediapipe"].get(manifest["sample"], {}))
        rt_result = sample_errors(manifest, methods["rtmpose"].get(manifest["sample"], {}))
        mp_nme = float(mp_result[1].mean()) if mp_result is not None else math.inf
        rt_nme = float(rt_result[1].mean()) if rt_result is not None else math.inf
        scored.append((manifest, mp_nme, rt_nme))

    buckets = [
        sorted(
            [row for row in scored if not math.isfinite(row[1]) and math.isfinite(row[2])],
            key=lambda row: row[2],
        ),
        sorted(
            [row for row in scored if math.isfinite(row[1]) and math.isfinite(row[2])],
            key=lambda row: row[1] - row[2],
            reverse=True,
        ),
        sorted(
            [row for row in scored if math.isfinite(row[1]) and math.isfinite(row[2])],
            key=lambda row: row[2] - row[1],
            reverse=True,
        ),
        sorted(
            [row for row in scored if math.isfinite(row[2])],
            key=lambda row: row[2],
            reverse=True,
        ),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_bucket = max(1, math.ceil(limit / len(buckets)))
    for bucket in buckets:
        for manifest, _, _ in bucket[:per_bucket]:
            if manifest["sample"] not in seen:
                seen.add(manifest["sample"])
                selected.append(manifest)
                if len(selected) == limit:
                    return selected
    for manifest, _, _ in scored:
        if manifest["sample"] not in seen:
            selected.append(manifest)
            seen.add(manifest["sample"])
            if len(selected) == limit:
                break
    return selected


def render_summary_plot(
    output: Path,
    summaries: dict[str, dict[str, Any]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["mediapipe", "rtmpose"]
    colors = ["#E69F00", "#CC79A7"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(names, [summaries[n]["return_rate"] * 100 for n in names], color=colors)
    axes[0, 0].set_title("Prediction return rate")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].set_ylim(0, 105)
    for i, n in enumerate(names):
        axes[0, 0].text(i, summaries[n]["return_rate"] * 100 + 1, f"{summaries[n]['return_rate']*100:.1f}%", ha="center")

    axes[0, 1].bar(names, [summaries[n]["nme_median_per_sample_conditional"] for n in names], color=colors)
    axes[0, 1].set_title("Conditional median per-sample NME (lower is better)")
    axes[0, 1].set_ylabel("NME / GT hand-box side")

    for name, color in zip(names, colors):
        thresholds, curve = curves[name]
        axes[1, 0].plot(thresholds, curve, label=name, color=color, linewidth=2)
    axes[1, 0].set_title("Failure-aware PCK curve")
    axes[1, 0].set_xlabel("Normalized distance threshold")
    axes[1, 0].set_ylabel("PCK (missed frames contribute zero)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    x = np.arange(21)
    for name, color in zip(names, colors):
        axes[1, 1].plot(x, summaries[name]["per_joint_epe_px_conditional"], marker="o", markersize=3, label=name, color=color)
    axes[1, 1].set_title("Per-joint conditional EPE")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(JOINT_NAMES, rotation=65, ha="right", fontsize=8)
    axes[1, 1].set_ylabel("Pixels")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle("DexYCB test: MediaPipe vs RTMPose-m Hand5", fontsize=15)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def command_report(args: argparse.Namespace) -> None:
    manifests = jsonl_read(args.output_root / "manifest.jsonl")
    prepare_summary = json.loads((args.output_root / "prepare_summary.json").read_text())
    prediction_lists = {
        "mediapipe": jsonl_read(args.output_root / "mediapipe_predictions.jsonl"),
        "rtmpose": jsonl_read(args.output_root / "rtmpose_predictions.jsonl"),
    }
    if any(len(rows) != len(manifests) for rows in prediction_lists.values()):
        raise ValueError("manifest and prediction files have different lengths")

    summaries: dict[str, dict[str, Any]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rows in prediction_lists.items():
        metrics, thresholds, pck = evaluate_method(manifests, rows)
        summaries[name] = metrics
        curves[name] = (thresholds, pck)

    valid_manifests = [row for row in manifests if "error" not in row]
    evaluable_manifests = [
        row for row in valid_manifests
        if np.asarray(row["gt_valid"], dtype=bool).sum() >= 2
    ]
    detector_rows = [row.get("detector", {}) for row in valid_manifests]
    detector_ious = [
        float(row["bbox_iou_gt"]) for row in detector_rows if row.get("bbox_iou_gt") is not None
    ]
    bbox_source = prepare_summary["bbox_source"]
    bbox_description = (
        "same current YOLO bbox expanded 1.5x; full-frame fallback on detector miss"
        if bbox_source == "detector"
        else "same GT-keypoint tight bbox expanded 1.5x"
    )
    detector_report = None
    if bbox_source == "detector":
        detector_report = {
            "return_rate": sum(bool(row.get("returned")) for row in detector_rows) / max(len(detector_rows), 1),
            "mean_iou_gt": float(np.mean(detector_ious)) if detector_ious else None,
            "recall_iou_0.5": sum(value >= 0.5 for value in detector_ious) / max(len(valid_manifests), 1),
        }
    report = {
        "protocol": {
            "dataset": "DexYCB canonical-right full-resolution",
            "split": "official s0_test clean list",
            "bbox_source": bbox_source,
            "bbox": bbox_description,
            "gt_visibility": "finite projected joints inside the original image",
            "normalizer": "maximum side of tight all-finite projected-GT-joint bbox",
            "failure_policy": "missing frames count as zero-correct in PCK/AUC",
            "conditional_policy": "EPE/NME only use frames where a method returned 21 finite points",
        },
        "selected_samples": len(manifests),
        "loaded_samples": len(valid_manifests),
        "evaluable_samples": len(evaluable_manifests),
        "detector": detector_report,
        "methods": summaries,
    }
    json_dump(args.output_root / "comparison_summary.json", report)
    render_summary_plot(args.output_root / "comparison_summary.png", summaries, curves)

    by_method = {
        name: {row["sample"]: row for row in rows}
        for name, rows in prediction_lists.items()
    }
    selected = select_visual_samples(evaluable_manifests, by_method, args.visual_samples)
    image_dir = args.output_root / "visualizations"
    image_dir.mkdir(parents=True, exist_ok=True)
    comparison_images = []
    for index, manifest in enumerate(selected):
        methods = {
            name: by_method[name].get(manifest["sample"], {"returned": False})
            for name in by_method
        }
        image = render_sample_comparison(manifest, methods)
        output = image_dir / f"{index:03d}_{Path(manifest['sample']).name}.jpg"
        cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        comparison_images.append(image)
    make_contact_sheet(comparison_images, args.output_root / "comparison_contact_sheet.jpg")

    lines = [
        "# DexYCB 手部关键点检测对比",
        "",
        f"选取 {report['selected_samples']} 张，2D 精度可评测 {report['evaluable_samples']} 张；共同裁框：{bbox_description}。",
        "缺失预测在总体 PCK/AUC 中计为错误；EPE/NME 为成功返回 21 点后的条件精度。",
        "",
        "| 方法 | 返回率 | 条件 EPE(px) | 条件 NME | PCK@0.1(总体) | PCK@0.2(总体) | AUC@0.5(总体) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("mediapipe", "rtmpose"):
        row = summaries[name]
        lines.append(
            f"| {name} | {row['return_rate']*100:.2f}% | "
            f"{row['epe_px_mean_conditional']:.3f} | {row['nme_mean_conditional']:.4f} | "
            f"{row['pck_0.10_all']:.4f} | {row['pck_0.20_all']:.4f} | "
            f"{row['auc_0.5_all']:.4f} |"
        )
    if detector_report is not None:
        lines += [
            "",
            "## 共享 detector",
            "",
            f"- 返回率：{detector_report['return_rate']*100:.2f}%",
            f"- 与 GT 关键点框平均 IoU：{detector_report['mean_iou_gt']:.4f}",
            f"- IoU≥0.5 recall：{detector_report['recall_iou_0.5']*100:.2f}%",
        ]
    lines += ["", "完整定义和逐关节结果见 `comparison_summary.json`。"]
    (args.output_root / "comparison_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"outputs: {args.output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    prepare.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES_FILE)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare.add_argument("--max-samples", type=int, default=1000)
    prepare.add_argument("--sampling", choices=("uniform", "random", "head"), default="uniform")
    prepare.add_argument("--seed", type=int, default=20260915)
    prepare.add_argument("--bbox-source", choices=("detector", "gt"), default="detector")
    prepare.add_argument("--bbox-expand", type=float, default=1.5)
    prepare.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR)
    prepare.add_argument("--detector-conf", type=float, default=0.25)
    prepare.add_argument("--detector-iou", type=float, default=0.7)
    prepare.add_argument("--detector-device", default="0")
    prepare.add_argument("--mediapipe-input-size", type=int, default=224)
    prepare.add_argument("--mediapipe-min-detection-confidence", type=float, default=0.3)
    prepare.add_argument("--progress-every", type=int, default=100)
    prepare.set_defaults(func=command_prepare)

    rtmpose = subparsers.add_parser("rtmpose")
    rtmpose.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    rtmpose.add_argument("--rtmpose-config", type=Path, default=DEFAULT_RTMPOSE_CONFIG)
    rtmpose.add_argument("--rtmpose-checkpoint", type=Path, default=DEFAULT_RTMPOSE_CHECKPOINT)
    rtmpose.add_argument("--device", default="cuda:0")
    rtmpose.add_argument("--batch-size", type=int, default=64)
    rtmpose.set_defaults(func=command_rtmpose)

    report = subparsers.add_parser("report")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument("--visual-samples", type=int, default=16)
    report.set_defaults(func=command_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Run MediaPipe Hands on HUG-converted DexYCB and HO3D samples.

The converted PKL files are used deliberately: their embedded RGB image is the
same 224x224 input consumed by the current hand-reconstruction pipeline, and it
already includes DexYCB left-to-right canonicalization where applicable.

Outputs per dataset:
  images/*.jpg       landmark overlays, including explicit failure images
  detections.jsonl   one record per sample with all detected 21-point hands
  summary.json       aggregate detection statistics and run configuration
  contact_sheet.jpg  compact visual overview of the sampled results

Example:
  conda run -n hug_mediapipe python -m src.detect_mediapipe_keypoints \
      --datasets dexycb_train ho3d_eval --max-samples 64

Use ``--max-samples 0 --no-save-images`` to process the complete sample lists
without creating tens of thousands of visualization images.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np


DEFAULT_OUTPUT_ROOT = Path("/root/code/vepfs/HUG-for-Recon-Gen/mediapipe_keypoints")
DEFAULT_DATASETS = {
    "dexycb_train": {
        "root": Path(
            "/root/code/vepfs/dataset/hand_recon_hug/"
            "dexycb_v2_canonical_right"
        ),
        "samples": Path(
            "/root/code/vepfs/dataset/hand_recon_hug/"
            "splits_v2/dexycb_train.clean.txt"
        ),
    },
    "dexycb_test": {
        "root": Path(
            "/root/code/vepfs/dataset/hand_recon_hug/"
            "dexycb_v2_canonical_right"
        ),
        "samples": Path(
            "/root/code/vepfs/dataset/hand_recon_hug/"
            "splits_v2/dexycb_test.clean.txt"
        ),
    },
    "ho3d_train": {
        "root": Path("/root/code/vepfs/dataset/hand_recon_hug/ho3d"),
        "samples": Path(
            "/root/code/vepfs/dataset/hand_recon_hug/"
            "splits/ho3d_train.clean.txt"
        ),
    },
    "ho3d_eval": {
        "root": Path("/root/code/vepfs/dataset/hand_recon_hug/ho3d_eval"),
        "samples": Path(
            "/root/code/vepfs/dataset/hand_recon_hug/"
            "splits_v2/ho3d_eval.clean.txt"
        ),
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    samples: Path


@dataclass
class DatasetSummary:
    dataset: str
    dataset_root: str
    samples_file: str
    available_samples: int
    processed_samples: int
    detected_samples: int
    failed_samples: int
    detection_rate: float
    detected_hands: int
    elapsed_seconds: float
    samples_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and visualize MediaPipe hand landmarks on DexYCB/HO3D."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_DATASETS),
        default=["dexycb_train", "ho3d_eval"],
        help="Dataset splits to process (default: dexycb_train ho3d_eval).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Samples per dataset; 0 processes the complete sample list.",
    )
    parser.add_argument(
        "--sampling",
        choices=("uniform", "random", "head"),
        default="uniform",
        help="How to select samples when --max-samples is positive.",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-num-hands", type=int, default=1)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--input-mirrored",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether images are selfie-mirrored. MediaPipe handedness assumes "
            "mirrored input; dataset images default to non-mirrored."
        ),
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save every overlay image, including detection failures.",
    )
    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=64,
        help="Maximum overlay images included in each contact sheet; 0 disables it.",
    )
    parser.add_argument("--dexycb-root", type=Path)
    parser.add_argument("--dexycb-samples", type=Path)
    parser.add_argument("--ho3d-root", type=Path)
    parser.add_argument("--ho3d-samples", type=Path)
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for name in args.datasets:
        defaults = DEFAULT_DATASETS[name]
        family = name.split("_", maxsplit=1)[0]
        root = getattr(args, f"{family}_root") or defaults["root"]
        samples = getattr(args, f"{family}_samples") or defaults["samples"]
        if not root.is_dir():
            raise FileNotFoundError(f"{name} dataset root not found: {root}")
        if not samples.is_file():
            raise FileNotFoundError(f"{name} samples file not found: {samples}")
        specs.append(DatasetSpec(name=name, root=root, samples=samples))
    return specs


def read_stems(samples_file: Path) -> list[str]:
    stems = [line.strip() for line in samples_file.read_text().splitlines()]
    return [stem for stem in stems if stem and not stem.startswith("#")]


def select_stems(
    stems: Sequence[str], max_samples: int, mode: str, seed: int
) -> list[str]:
    if max_samples <= 0 or max_samples >= len(stems):
        return list(stems)
    if mode == "head":
        return list(stems[:max_samples])
    if mode == "random":
        indices = sorted(random.Random(seed).sample(range(len(stems)), max_samples))
        return [stems[index] for index in indices]
    indices = np.linspace(0, len(stems) - 1, num=max_samples, dtype=np.int64)
    return [stems[int(index)] for index in indices]


def load_rgb(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as handle:
        sample = pickle.load(handle)
    image_bytes = sample.get("image")
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError(f"missing encoded image bytes in {pkl_path}")
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode image in {pkl_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def landmarks_to_record(
    hand_landmarks: Any,
    handedness: Any | None,
    width: int,
    height: int,
    input_mirrored: bool,
) -> dict[str, Any]:
    if handedness is not None and handedness.classification:
        classification = handedness.classification[0]
        raw_label = classification.label
        score = float(classification.score)
    else:
        raw_label = "Unknown"
        score = 0.0
    if input_mirrored:
        label = raw_label
    else:
        label = {"Left": "Right", "Right": "Left"}.get(raw_label, raw_label)

    normalized = []
    pixels = []
    for landmark in hand_landmarks.landmark:
        x, y, z = float(landmark.x), float(landmark.y), float(landmark.z)
        normalized.append([x, y, z])
        pixels.append([x * width, y * height])

    xy = np.asarray(pixels, dtype=np.float32)
    bbox_xyxy = [
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    ]
    return {
        "handedness": label,
        "mediapipe_raw_handedness": raw_label,
        "handedness_score": score,
        "landmarks_normalized_xyz": normalized,
        "landmarks_pixel_xy": pixels,
        "landmark_bbox_xyxy": bbox_xyxy,
    }


def draw_result(
    rgb: np.ndarray,
    hand_landmarks: Sequence[Any],
    handedness: Sequence[Any],
    sample_name: str,
    input_mirrored: bool,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    draw = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    for index, landmarks in enumerate(hand_landmarks):
        draw.draw_landmarks(
            canvas,
            landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            styles.get_default_hand_landmarks_style(),
            styles.get_default_hand_connections_style(),
        )
        handed = handedness[index] if index < len(handedness) else None
        if handed is not None and handed.classification:
            cls = handed.classification[0]
            corrected = (
                cls.label
                if input_mirrored
                else {"Left": "Right", "Right": "Left"}.get(cls.label, cls.label)
            )
            label = f"hand {index}: {corrected} {cls.score:.2f}"
        else:
            label = f"hand {index}"
        points = np.asarray(
            [[lm.x * canvas.shape[1], lm.y * canvas.shape[0]] for lm in landmarks.landmark]
        )
        anchor = points.min(axis=0).astype(int)
        cv2.putText(
            canvas,
            label,
            (max(2, int(anchor[0])), max(16, int(anchor[1]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 255, 40),
            1,
            cv2.LINE_AA,
        )

    detected = bool(hand_landmarks)
    status = f"DETECTED ({len(hand_landmarks)})" if detected else "NO HAND DETECTED"
    color = (30, 210, 30) if detected else (20, 20, 235)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 39), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        status,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        sample_name[-52:],
        (6, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(
    image_paths: Sequence[Path], output_path: Path, columns: int = 8
) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    tile_width, tile_height = 256, 256
    rows = math.ceil(len(images) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 242, np.uint8)
    for index, image in enumerate(images):
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        new_size = (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        )
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x0 = column * tile_width + (tile_width - new_size[0]) // 2
        y0 = row * tile_height + (tile_height - new_size[1]) // 2
        sheet[y0 : y0 + new_size[1], x0 : x0 + new_size[0]] = resized
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def process_dataset(
    spec: DatasetSpec,
    hands: Any,
    output_root: Path,
    max_samples: int,
    sampling: str,
    seed: int,
    save_images: bool,
    contact_sheet_limit: int,
    input_mirrored: bool,
) -> DatasetSummary:
    all_stems = read_stems(spec.samples)
    stems = select_stems(all_stems, max_samples, sampling, seed)
    dataset_out = output_root / spec.name
    images_out = dataset_out / "images"
    dataset_out.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_out.mkdir(parents=True, exist_ok=True)

    detected_samples = 0
    detected_hands = 0
    records: list[dict[str, Any]] = []
    contact_paths: list[Path] = []
    started = time.perf_counter()

    for index, stem in enumerate(stems):
        pkl_path = spec.root / f"{stem}.pkl"
        record: dict[str, Any] = {
            "dataset": spec.name,
            "sample": stem,
            "pkl_path": str(pkl_path),
            "detected": False,
            "hands": [],
        }
        try:
            rgb = load_rgb(pkl_path)
            height, width = rgb.shape[:2]
            rgb_input = np.ascontiguousarray(rgb)
            rgb_input.flags.writeable = False
            result = hands.process(rgb_input)
            landmarks = list(result.multi_hand_landmarks or [])
            handedness = list(result.multi_handedness or [])
            record["image_size_wh"] = [width, height]
            record["hands"] = [
                landmarks_to_record(
                    hand,
                    handedness[hand_index] if hand_index < len(handedness) else None,
                    width,
                    height,
                    input_mirrored,
                )
                for hand_index, hand in enumerate(landmarks)
            ]
            record["detected"] = bool(landmarks)
            if landmarks:
                detected_samples += 1
                detected_hands += len(landmarks)

            if save_images:
                canvas = draw_result(
                    rgb, landmarks, handedness, stem, input_mirrored
                )
                safe_stem = Path(stem).name.replace("/", "__")
                image_path = images_out / f"{index:05d}_{safe_stem}.jpg"
                if not cv2.imwrite(
                    str(image_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]
                ):
                    raise OSError(f"failed to save {image_path}")
                record["visualization"] = str(image_path)
                if len(contact_paths) < contact_sheet_limit:
                    contact_paths.append(image_path)
        except Exception as error:  # keep long dataset runs auditable
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)

        if (index + 1) % 1000 == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[{spec.name}] {index + 1}/{len(stems)} "
                f"detected={detected_samples} rate={detected_samples/(index+1):.3f} "
                f"speed={(index+1)/elapsed:.1f} samples/s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    manifest_path = dataset_out / "detections.jsonl"
    with manifest_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = DatasetSummary(
        dataset=spec.name,
        dataset_root=str(spec.root),
        samples_file=str(spec.samples),
        available_samples=len(all_stems),
        processed_samples=len(records),
        detected_samples=detected_samples,
        failed_samples=len(records) - detected_samples,
        detection_rate=(detected_samples / len(records)) if records else 0.0,
        detected_hands=detected_hands,
        elapsed_seconds=elapsed,
        samples_per_second=(len(records) / elapsed) if elapsed > 0 else 0.0,
    )
    (dataset_out / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n"
    )
    if save_images and contact_sheet_limit > 0:
        make_contact_sheet(contact_paths, dataset_out / "contact_sheet.jpg")
    return summary


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[DatasetSummary] = []
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_num_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        for dataset_index, spec in enumerate(specs):
            summary = process_dataset(
                spec=spec,
                hands=hands,
                output_root=args.output_root,
                max_samples=args.max_samples,
                sampling=args.sampling,
                seed=args.seed + dataset_index,
                save_images=args.save_images,
                contact_sheet_limit=args.contact_sheet_limit,
                input_mirrored=args.input_mirrored,
            )
            summaries.append(summary)
            print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))

    run_summary = {
        "mediapipe_version": mp.__version__,
        "configuration": {
            "datasets": args.datasets,
            "max_samples_per_dataset": args.max_samples,
            "sampling": args.sampling,
            "seed": args.seed,
            "max_num_hands": args.max_num_hands,
            "model_complexity": args.model_complexity,
            "input_mirrored": args.input_mirrored,
            "min_detection_confidence": args.min_detection_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "save_images": args.save_images,
        },
        "datasets": [asdict(summary) for summary in summaries],
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"outputs: {args.output_root}")


if __name__ == "__main__":
    main()

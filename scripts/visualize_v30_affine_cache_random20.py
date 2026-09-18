"""Create reproducible random visualizations for affine/cache alignment."""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.dataloader.augmented_grasp_dataset import AugmentedGraspDataset
from src.dataloader.grasp_dataset import GraspDataset


CONFIG = Path("configs/train_handrecon_v30_native_rgbd.yaml")
OUTPUT = Path(
    "/root/code/vepfs/HUG-for-Recon-Gen/visualizations/"
    "v30_affine_cache_random20"
)
SELECTION_SEED = 20260917
COUNT = 20

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Diagnostic visualization override. The checked-in v30 baseline remains off.
AUGMENTATION = {
    "enabled": True,
    "color_scale": 0.10,
    "brightness_delta": 0.05,
    "brightness_prob": 1.0,
    "contrast_min": 0.90,
    "contrast_max": 1.10,
    "contrast_prob": 1.0,
    "depth": {"enabled": False},
    "pointcloud": {"enabled": False},
    "affine": {
        "enabled": True,
        "probability": 1.0,
        "scale_factor": 0.15,
        "rotation_deg": 15.0,
        "translation_frac": 0.08,
        "min_visible_landmarks": 0.95,
    },
}


def put_label(image, text, origin, color=(255, 255, 255), scale=0.65):
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )
    cv2.rectangle(
        image,
        (x - 4, y - height - 5),
        (x + width + 4, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_condition(image, condition, confidence_threshold, anchor=None):
    out = np.asarray(image, dtype=np.uint8).copy()
    detector = np.rint(condition["detector_bbox_xyxy"]).astype(int)
    crop = np.rint(condition["crop_bbox_xyxy"]).astype(int)
    if bool(condition["detector_hit"]):
        cv2.rectangle(out, detector[:2], detector[2:], (255, 0, 0), 3)
    cv2.rectangle(out, crop[:2], crop[2:], (0, 255, 0), 3)

    points = np.asarray(condition["keypoints_xy"], np.float32)
    scores = np.asarray(condition["keypoint_scores"], np.float32)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(scores)
        & (scores >= confidence_threshold)
    )
    for start, end in SKELETON_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                out,
                tuple(np.rint(points[start]).astype(int)),
                tuple(np.rint(points[end]).astype(int)),
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    for point in points[valid]:
        cv2.circle(
            out,
            tuple(np.rint(point).astype(int)),
            4,
            (255, 255, 0),
            -1,
            cv2.LINE_AA,
        )
    if anchor is not None and np.isfinite(anchor).all():
        center = tuple(np.rint(anchor).astype(int))
        cv2.drawMarker(
            out,
            center,
            (0, 255, 255),
            cv2.MARKER_DIAMOND,
            18,
            3,
            cv2.LINE_AA,
        )
    return out, int(valid.sum())


def draw_crop(image, points, valid, anchor):
    out = cv2.resize(image, (480, 480), interpolation=cv2.INTER_LINEAR)
    ratio = 480.0 / float(image.shape[0])
    points = np.asarray(points, np.float32) * ratio
    valid = np.asarray(valid, bool)
    for start, end in SKELETON_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                out,
                tuple(np.rint(points[start]).astype(int)),
                tuple(np.rint(points[end]).astype(int)),
                (255, 255, 0),
                3,
                cv2.LINE_AA,
            )
    for point in points[valid]:
        cv2.circle(
            out,
            tuple(np.rint(point).astype(int)),
            6,
            (255, 255, 0),
            -1,
            cv2.LINE_AA,
        )
    if anchor is not None and np.isfinite(anchor).all():
        center = tuple(np.rint(np.asarray(anchor) * ratio).astype(int))
        cv2.drawMarker(
            out,
            center,
            (0, 255, 255),
            cv2.MARKER_DIAMOND,
            22,
            4,
            cv2.LINE_AA,
        )
    cv2.rectangle(out, (1, 1), (478, 478), (0, 255, 0), 3)
    return out


def compose_triptych(original, augmented, crop, title, params, visible):
    header_height = 78
    canvas = np.zeros((480 + header_height, 1760, 3), dtype=np.uint8)
    canvas[header_height:, :640] = original
    canvas[header_height:, 640:1280] = augmented
    canvas[header_height:, 1280:] = crop
    put_label(canvas, title, (12, 27), scale=0.70)
    detail = (
        f"scale={params['scale']:.3f}  angle={params['angle_deg']:+.2f} deg  "
        f"translation=({params['tx_frac']:+.3f}W, {params['ty_frac']:+.3f}H)  "
        f"valid joints={visible}/21"
    )
    put_label(canvas, detail, (12, 61), scale=0.57)
    put_label(canvas, "ORIGINAL + CACHED CONDITION", (14, 107), scale=0.58)
    put_label(canvas, "AUGMENTED + TRANSFORMED CACHE", (654, 107), scale=0.58)
    put_label(canvas, "MODEL RGB CROP (224 -> 480)", (1294, 107), scale=0.58)
    return canvas


def make_dataset():
    cfg = OmegaConf.load(CONFIG)
    data_cfg = cfg.trainer.data
    model_cfg = cfg.trainer.model
    entry = data_cfg.datasets[0]
    hand_crop = OmegaConf.to_container(data_cfg.hand_crop, resolve=True)
    # Keep every RTMPose point visible for this coordinate diagnostic.
    hand_crop["skeleton_drop_prob"] = 0.0
    dataset = AugmentedGraspDataset(
        dataset_path=entry.path,
        split="train",
        samples_filename=entry.train_samples,
        image_size=int(model_cfg.image_size),
        use_rgb=True,
        use_depth=True,
        n_points_input=int(data_cfg.n_points_input),
        pcl_crop_radius=float(model_cfg.pcl_crop_radius),
        d_mano=int(model_cfg.d_mano),
        query_min_depth=float(data_cfg.query_min_depth),
        query_max_depth=float(data_cfg.query_max_depth),
        query_depth_cluster_width=float(data_cfg.query_depth_cluster_width),
        hand_crop=hand_crop,
        geometry_overlay=data_cfg.geometry_overlay,
        augmentation=AUGMENTATION,
    )
    return dataset, cfg


def build_readme(records, skipped, cfg):
    rows = []
    for record in records:
        params = record["affine_params"]
        rows.append(
            "| {number:02d} | `{file}` | `{stem}` | {index} | {seed} | "
            "{scale:.3f} | {angle:+.2f} | ({tx:+.3f}, {ty:+.3f}) | "
            "{visible}/21 | {side} |".format(
                number=record["number"],
                file=record["file"],
                stem=record["stem"],
                index=record["dataset_index"],
                seed=record["augmentation_seed"],
                scale=params["scale"],
                angle=params["angle_deg"],
                tx=params["tx_frac"],
                ty=params["ty_frac"],
                visible=record["valid_keypoints"],
                side=record["source_side"],
            )
        )

    return f"""# v30 affine 与离线 condition cache 随机 20 帧可视化

## 用途与数据

本目录用于检查数据增强后，离线 detector/RTMPose cache 是否与 RGB、Depth 和相机坐标保持一致。数据来自 v30 DexYCB train 列表，基础配置为 `{CONFIG}`。使用固定选择种子 `{SELECTION_SEED}` 从 {len(records) + skipped} 个随机候选中保留 20 个成功应用 affine 且 detector/RTMPose 有效的样本；跳过 {skipped} 个不满足条件的候选。因此结果可以复现，但不是按数据集顺序挑选。

`overview.jpg` 是 20 个增强整图的 5x4 总览；`samples/` 中每张三联图依次为：原始整图和原 cache、增强整图和同步变换后的 cache、模型实际接收的 224x224 RGB crop（显示时放大到 480x480）。`manifest.json` 保存精确样本名、索引、随机种子和每帧 affine 参数。

## 本次开启的增强

- **全图 affine**：100% 采样；缩放范围 0.85-1.15，旋转范围 -15 至 +15 度，水平和垂直平移范围均为图像尺寸的 -8% 至 +8%。要求至少 95% GT 关键点仍在画面内；不满足时该次 affine 会回退，本目录只保留实际应用成功的样本。
- **RGB 通道缩放**：每个颜色通道独立乘以 0.9-1.1。
- **亮度**：100% 采样，加性偏移范围为归一化 RGB 的 -0.05 至 +0.05。
- **对比度**：100% 采样，范围为 0.9-1.1。

本次没有开启 Depth 噪声和 point-cloud 扰动，因为这组图专门检查二维 affine 与 condition cache 的坐标同步。可视化时还将 `skeleton_drop_prob` 临时设为 0，避免随机丢弃整套骨架掩盖坐标检查。仓库中的 v30 基线配置仍保持 `augmentation.enabled: false`，本目录的参数是可视化诊断覆盖，不代表已经修改训练配置。

## 图中元素

- **红色框**：detector 原始 bbox。增强图中该框由四个角共同经过 affine 后重新取轴对齐包围框。
- **绿色框**：最终 RGB crop bbox。它根据变换后的红框按 `hand_crop.expand=1.5` 重新生成；右侧 crop 的绿色边框表示该区域被缩放为模型的 224x224 RGB 输入。
- **黄色点和连线**：confidence >= {cfg.trainer.data.hand_crop.rtmpose_min_joint_confidence} 的 RTMPose 21 点及骨架连接。增强整图和右侧 crop 使用同一组同步变换后的点。
- **青色菱形**：模型用于查询/深度定位的 palm anchor，即输出 `point_uv` 的二维位置。
- 黑色区域可能来自旋转/平移后的图像边界填充，属于 affine 的正常结果。

正确的对齐关系是：中间图的黄色骨架跟随手部；红框包围检测到的手；绿框覆盖完整手部；右侧 crop 内的黄色骨架与同一只手保持一致。

## 随机样本与参数

| # | 文件 | sample stem | dataset index | aug seed | scale | angle deg | translation (W,H) | valid joints | source side |
|---:|---|---|---:|---:|---:|---:|---|---:|---|
{chr(10).join(rows)}
"""


def main():
    dataset, cfg = make_dataset()
    sample_dir = OUTPUT / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SELECTION_SEED)
    candidates = rng.permutation(len(dataset))
    records = []
    overview_images = []
    attempts = 0

    for index in candidates:
        if len(records) >= COUNT:
            break
        attempts += 1
        aug_seed = SELECTION_SEED + attempts
        sample = dataset.get_augmented_for_viz(int(index), seed=aug_seed)
        state = sample["augmentation_state"]
        if state["applied_affine_matrix"] is None:
            continue

        original_condition = GraspDataset._condition_for_sample(dataset, int(index))
        previous_state = dataset._augmentation_state
        dataset._augmentation_state = state
        try:
            augmented_condition = dataset._condition_for_sample(int(index))
        finally:
            dataset._augmentation_state = previous_state
        if not (
            bool(augmented_condition["detector_hit"])
            and bool(augmented_condition["pose_returned"])
        ):
            continue

        original_rgb = GraspDataset._decode_image(sample["raw_original"]["image"])
        augmented_rgb = sample["rgb_augmented"]
        anchor_full = sample["item"]["point_uv"][:2].numpy()
        original_drawn, _ = draw_condition(
            original_rgb,
            original_condition,
            dataset.rtmpose_min_joint_conf,
        )
        augmented_drawn, visible = draw_condition(
            augmented_rgb,
            augmented_condition,
            dataset.rtmpose_min_joint_conf,
            anchor=anchor_full,
        )

        crop_bbox = sample["item"]["hand_crop_bbox"].numpy()
        crop_affine = dataset._crop_affine(crop_bbox, dataset.image_size)
        crop_rgb = cv2.warpAffine(
            augmented_rgb,
            crop_affine,
            (dataset.image_size, dataset.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        anchor_crop = dataset._transform_points(anchor_full[None], crop_affine)[0]
        crop_drawn = draw_crop(
            crop_rgb,
            sample["item"]["hand_keypoints_2d"].numpy(),
            sample["item"]["hand_keypoints_valid"].numpy(),
            anchor_crop,
        )

        number = len(records) + 1
        stem = str(sample["item"]["stem"])
        short_stem = "/".join(stem.split("/")[-4:])
        side = "left" if bool(sample["item"]["source_is_left"].item()) else "right"
        filename = f"sample_{number:02d}.jpg"
        params = {key: float(value) for key, value in state["affine_params"].items()}
        triptych = compose_triptych(
            original_drawn,
            augmented_drawn,
            crop_drawn,
            f"#{number:02d}  {short_stem}  source={side}",
            params,
            visible,
        )
        if not cv2.imwrite(
            str(sample_dir / filename), cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"failed to write {filename}")

        thumb = cv2.resize(augmented_drawn, (320, 240), interpolation=cv2.INTER_AREA)
        put_label(
            thumb,
            f"#{number:02d}  {side.upper()}  angle={params['angle_deg']:+.1f} deg",
            (7, 24),
            scale=0.42,
        )
        overview_images.append(thumb)
        records.append(
            {
                "number": number,
                "file": f"samples/{filename}",
                "stem": stem,
                "dataset_index": int(index),
                "augmentation_seed": int(aug_seed),
                "source_side": side,
                "affine_params": params,
                "affine_matrix": np.asarray(
                    state["applied_affine_matrix"], np.float32
                ).tolist(),
                "detector_bbox_xyxy": np.asarray(
                    augmented_condition["detector_bbox_xyxy"], np.float32
                ).tolist(),
                "crop_bbox_xyxy": np.asarray(
                    augmented_condition["crop_bbox_xyxy"], np.float32
                ).tolist(),
                "valid_keypoints": visible,
                "anchor_uv": anchor_full.astype(float).tolist(),
            }
        )

    if len(records) != COUNT:
        raise RuntimeError(f"only produced {len(records)} of {COUNT} samples")

    rows = [np.concatenate(overview_images[i:i + 5], axis=1) for i in range(0, 20, 5)]
    overview = np.concatenate(rows, axis=0)
    if not cv2.imwrite(
        str(OUTPUT / "overview.jpg"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError("failed to write overview")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(CONFIG),
        "selection_seed": SELECTION_SEED,
        "requested_samples": COUNT,
        "random_candidates_examined": attempts,
        "candidates_skipped": attempts - COUNT,
        "visualization_overrides": {
            "augmentation": AUGMENTATION,
            "hand_crop.skeleton_drop_prob": 0.0,
        },
        "legend": {
            "red_box": "detector_bbox_xyxy",
            "green_box": "expanded RGB crop bbox",
            "yellow": "valid RTMPose keypoints and skeleton",
            "cyan_diamond": "palm/query anchor point_uv",
        },
        "samples": records,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        build_readme(records, attempts - COUNT, cfg), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "samples": len(records),
        "random_candidates_examined": attempts,
        "overview": str(OUTPUT / "overview.jpg"),
        "readme": str(OUTPUT / "README.md"),
        "manifest": str(OUTPUT / "manifest.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

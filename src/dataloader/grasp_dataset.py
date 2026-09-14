"""Grasp dataset for training and evaluation."""

import logging
import pickle
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.data_keys import MANO_RIGHT_MESH_FACES_FILE, MANO_RIGHT_SHAPE_FILE
from ..utils.pcl_utils import depth_to_pcl_tensors, pixel_to_xyz

logger = logging.getLogger(__name__)

# Prediction output dir, excluded from input pkl discovery
PRED_DIRNAME = "grasp_pred"

# HO3D_v3 evaluation_xyz.json stores joints in MANO's RAW kinematic order
# [wrist, index MCP/PIP/DIP, middle, pinky, ring, thumb CMC/MCP/IP, then tips
# (thumb, index, middle, ring, pinky)]. Everything in this repo (manotorch
# output, our 21-joint predictions) uses the reordered convention
# [wrist, thumb x4, index x4, middle x4, ring x4, pinky x4] - the same perm as
# manotorch/manolayer.py's final reorder. std[i] = raw[HO3D_RAW_TO_STD[i]].
# Verified empirically: per-joint bone-length signatures match only under
# this mapping (see TRAIN_HANDRECON.md).
HO3D_RAW_TO_STD = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class GraspDataset(Dataset):
    """Dataset for grasp prediction from RGB + object mask.

    Loads RGB images, object masks, and ground truth MANO parameters from any
    `.pkl` found recursively under dataset_path (the `grasp_pred/` output dir
    excluded). Output MANO pose: 99D = 3 (metric t in meters) +
    6 (wrist R_6d) + 90 (15 joints * 6D).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        indices: Optional[List[int]] = None,
        image_size: int = 224,
        use_rgb: bool = True,
        use_depth: bool = True,
        samples_filename: Optional[str] = None,
        n_points_input: int = 4096,
        pcl_crop_radius: Optional[float] = 0.3,
        d_mano: int = 99,
        query_min_depth: float = 0.15,
        query_max_depth: float = 2.0,
        query_depth_cluster_width: float = 0.12,
        hand_crop: Optional[Dict] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.n_points_input = n_points_input
        self.pcl_crop_radius = pcl_crop_radius
        self.query_min_depth = max(float(query_min_depth), 0.0)
        self.query_max_depth = float(query_max_depth)
        self.query_depth_cluster_width = max(float(query_depth_cluster_width), 0.0)
        self.hand_crop = dict(hand_crop or {})
        self.hand_crop_enabled = bool(self.hand_crop.get("enabled", False))
        self.hand_crop_expand = max(float(self.hand_crop.get("expand", 1.5)), 1.0)
        self.depth_radius_expand = max(
            float(self.hand_crop.get("depth_radius_expand", 1.25)), 1.0
        )
        self.detector_weights_path = self.hand_crop.get("detector_weights")
        self.detector_conf = float(self.hand_crop.get("detector_conf", 0.25))
        self.detector_iou = float(self.hand_crop.get("detector_iou", 0.7))
        self.detector_device = str(self.hand_crop.get("detector_device", "cpu"))
        self.mediapipe_enabled = bool(self.hand_crop.get("mediapipe_enabled", True))
        self.mediapipe_min_conf = float(
            self.hand_crop.get("mediapipe_min_detection_confidence", 0.3)
        )
        self.keypoint_source = str(
            self.hand_crop.get("keypoint_source", "mediapipe")
        ).strip().lower()
        if self.keypoint_source not in {"mediapipe", "gt"}:
            raise ValueError(
                "hand_crop.keypoint_source must be 'mediapipe' or 'gt', got "
                f"{self.keypoint_source!r}"
            )
        # Third-party models are created lazily inside the DataLoader process.
        # Keeping them out of __init__ makes the dataset safe to pickle/fork.
        self._hand_detector = None
        self._mediapipe_hands = None
        if self.query_max_depth <= self.query_min_depth:
            raise ValueError(
                "query_max_depth must be greater than query_min_depth, got "
                f"{self.query_min_depth}..{self.query_max_depth}"
            )
        self.d_mano = int(d_mano)
        if self.d_mano not in (99, 109):
            raise ValueError(
                f"d_mano must be 99 (legacy) or 109 (learnable MANO shape), got {self.d_mano}"
            )

        self.grasp_files = self._load_file_list(
            self.dataset_path, split, samples_filename
        )

        if indices is not None:
            self.grasp_files = [self.grasp_files[i] for i in indices]

        # Image transforms (ImageNet normalization for DINOv2)
        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.ToTensor()

    @staticmethod
    def _bbox_from_keypoints(keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Tight xyxy box around finite 2D joints."""
        xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        valid = np.isfinite(xy).all(axis=1)
        if valid.sum() < 2:
            return None
        xy = xy[valid]
        return np.array(
            [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()],
            dtype=np.float32,
        )

    @staticmethod
    def _expanded_square_bbox(
        bbox: np.ndarray, expand: float, min_side: float = 24.0
    ) -> np.ndarray:
        """Convert xyxy to an expanded square without clipping the padding."""
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        side = max(float(x2 - x1), float(y2 - y1), float(min_side)) * float(expand)
        half = 0.5 * side
        return np.array([cx - half, cy - half, cx + half, cy + half], np.float32)

    def _load_detector(self):
        if self._hand_detector is None:
            if not self.detector_weights_path:
                raise ValueError(
                    "hand_crop.detector_weights is required for val/test detector crop"
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "Detector crop requires ultralytics. Run this configuration in "
                    "the hug_mediapipe environment with ultralytics installed."
                ) from exc
            self._hand_detector = YOLO(str(self.detector_weights_path))
        return self._hand_detector

    def _detect_hand_bbox(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Highest-confidence right-hand YOLO box, falling back to any hand."""
        detector = self._load_detector()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = detector(
            bgr,
            conf=self.detector_conf,
            iou=self.detector_iou,
            device=self.detector_device,
            verbose=False,
        )[0]
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
        right = np.flatnonzero(classes > 0.5)
        candidates = right if right.size else np.arange(len(scores))
        return boxes[int(candidates[np.argmax(scores[candidates])])]

    def _select_hand_crop_bbox(
        self, grasp_data: Dict, rgb: np.ndarray
    ) -> tuple[np.ndarray, bool, str]:
        """Use GT joints in train and the requested detector in val/test."""
        height, width = rgb.shape[:2]
        if self.split == "train":
            grasp = grasp_data.get("grasp")
            bbox = self._bbox_from_keypoints(grasp.get("landmarks_2d")) if grasp else None
            source = "gt"
        else:
            bbox = self._detect_hand_bbox(rgb)
            source = "detector"
        if bbox is None:
            # A miss must not invalidate the sample or couple all modalities to
            # detector failure. Full-frame local input is the deterministic fallback.
            return np.array([0.0, 0.0, float(width - 1), float(height - 1)]), False, "full_fallback"
        return self._expanded_square_bbox(bbox, self.hand_crop_expand), True, source

    @staticmethod
    def _crop_affine(bbox: np.ndarray, output_size: int) -> np.ndarray:
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4)
        sx = float(output_size - 1) / max(float(x2 - x1), 1.0)
        sy = float(output_size - 1) / max(float(y2 - y1), 1.0)
        return np.array([[sx, 0.0, -sx * x1], [0.0, sy, -sy * y1]], np.float32)

    @staticmethod
    def _transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
        xy = np.asarray(points, dtype=np.float32)
        shape = xy.shape
        flat = xy.reshape(-1, shape[-1]).copy()
        xy1 = np.concatenate(
            [flat[:, :2], np.ones((len(flat), 1), dtype=np.float32)], axis=1
        )
        flat[:, :2] = xy1 @ affine.T
        return flat.reshape(shape)

    @staticmethod
    def _inverse_transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
        homography = np.eye(3, dtype=np.float32)
        homography[:2] = affine
        inverse = np.linalg.inv(homography).astype(np.float32)[:2]
        return GraspDataset._transform_points(points, inverse)

    def _crop_rgb(
        self,
        rgb: np.ndarray,
        K: np.ndarray,
        bbox: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop only RGB and return its crop-space camera intrinsics."""
        affine = self._crop_affine(bbox, self.image_size)
        size = (self.image_size, self.image_size)
        rgb_crop = cv2.warpAffine(
            rgb,
            affine,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        homography = np.eye(3, dtype=np.float32)
        homography[:2] = affine
        K_crop = homography @ np.asarray(K, dtype=np.float32)
        return rgb_crop, K_crop, affine

    def _crop_original_depth_by_keypoint_radius(
        self,
        depth_m: torch.Tensor,
        keypoints_original: np.ndarray,
        keypoints_valid: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray, float]:
        """Mask the original depth by a keypoint-derived image-space circle."""
        depth = depth_m.numpy() if isinstance(depth_m, torch.Tensor) else np.asarray(depth_m)
        valid = np.asarray(keypoints_valid, dtype=bool)
        points = np.asarray(keypoints_original, dtype=np.float32)[valid]
        if len(points) < 2:
            raise ValueError("keypoint-radius depth crop requires at least two joints")
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = float(np.linalg.norm(points - center[None], axis=1).max())
        radius = max(radius * self.depth_radius_expand, 8.0)
        yy, xx = np.ogrid[: depth.shape[0], : depth.shape[1]]
        mask = (xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2 <= radius ** 2
        cropped = np.where(mask, depth, 0.0).astype(np.float32, copy=False)
        return torch.from_numpy(cropped), center.astype(np.float32), radius

    def _load_mediapipe(self):
        if self._mediapipe_hands is None:
            try:
                import mediapipe as mp
            except ImportError as exc:
                raise ImportError(
                    "MediaPipe skeleton conditioning requires mediapipe. "
                    "Use the hug_mediapipe environment."
                ) from exc
            self._mediapipe_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=self.mediapipe_min_conf,
                min_tracking_confidence=0.5,
            )
        return self._mediapipe_hands

    def _mediapipe_keypoints(self, rgb_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return crop-pixel landmarks and a per-joint validity mask."""
        zeros_xy = np.zeros((21, 2), dtype=np.float32)
        zeros_valid = np.zeros(21, dtype=np.bool_)
        if not self.mediapipe_enabled:
            return zeros_xy, zeros_valid
        result = self._load_mediapipe().process(np.ascontiguousarray(rgb_crop))
        if not result.multi_hand_landmarks:
            return zeros_xy, zeros_valid
        landmarks = result.multi_hand_landmarks[0].landmark
        xy = np.asarray(
            [[lm.x * self.image_size, lm.y * self.image_size] for lm in landmarks],
            dtype=np.float32,
        )
        valid = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0.0)
            & (xy[:, 0] < self.image_size)
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] < self.image_size)
        )
        return np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0), valid

    def _hand_keypoints(
        self,
        grasp_data: Dict,
        rgb_crop: np.ndarray,
        rgb_original: np.ndarray,
        crop_affine: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return crop-space skeleton points and full-frame points for Depth.

        Oracle A replaces only the MediaPipe landmark source with GT. The GT
        joints are transformed into the detector/GT RGB crop for the skeleton
        condition, while their original full-frame coordinates define the
        adaptive Depth ROI. Separate validity masks preserve this distinction:
        a detector box can exclude a GT joint from the RGB crop without also
        deleting that joint from the oracle Depth ROI.
        """
        if self.keypoint_source == "mediapipe":
            keypoints_crop, valid_crop = self._mediapipe_keypoints(rgb_crop)
            keypoints_original = self._inverse_transform_points(
                keypoints_crop, crop_affine
            )
            return keypoints_crop, valid_crop, keypoints_original, valid_crop.copy()

        grasp = grasp_data.get("grasp")
        landmarks = grasp.get("landmarks_2d") if isinstance(grasp, dict) else None
        if landmarks is None:
            raise ValueError(
                "hand_crop.keypoint_source='gt' requires grasp.landmarks_2d; "
                "Oracle A is only valid on labeled reconstruction samples"
            )
        keypoints_original = np.asarray(landmarks, dtype=np.float32)
        if keypoints_original.shape != (21, 2):
            raise ValueError(
                "Oracle A expects grasp.landmarks_2d with shape (21, 2), got "
                f"{keypoints_original.shape}"
            )

        # DexYCB includes valid projections outside the image for partially
        # visible hands. Keep every finite GT joint: the expanded RGB crop can
        # include padded image regions, and the full-frame Depth circle is
        # naturally clipped by the depth image boundary.
        valid_original = np.isfinite(keypoints_original).all(axis=1)
        keypoints_crop = self._transform_points(keypoints_original, crop_affine)
        valid_crop = (
            valid_original
            & np.isfinite(keypoints_crop).all(axis=1)
            & (keypoints_crop[:, 0] >= 0.0)
            & (keypoints_crop[:, 0] < self.image_size)
            & (keypoints_crop[:, 1] >= 0.0)
            & (keypoints_crop[:, 1] < self.image_size)
        )
        keypoints_crop = np.nan_to_num(
            keypoints_crop, nan=0.0, posinf=0.0, neginf=0.0
        )
        keypoints_original = np.nan_to_num(
            keypoints_original, nan=0.0, posinf=0.0, neginf=0.0
        )
        return keypoints_crop, valid_crop, keypoints_original, valid_original

    def _palm_anchor(self, keypoints: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Stable 2D anchor from wrist and four MCP joints."""
        palm_ids = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
        keep = valid[palm_ids]
        if keep.any():
            return np.median(keypoints[palm_ids][keep], axis=0).astype(np.float32)
        return np.array([0.5 * (self.image_size - 1)] * 2, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.grasp_files)

    @staticmethod
    def find_pkls(root: Path) -> List[Path]:
        """All .pkl under root (recursive, sorted), excluding the grasp_pred/ output dir."""
        return sorted(
            p
            for p in root.rglob("*.pkl")
            if PRED_DIRNAME not in p.relative_to(root).parts
        )

    @staticmethod
    def _load_file_list(
        dataset_path: Path,
        split: str,
        samples_filename: Optional[str] = None,
    ) -> List[Path]:
        """Resolve list of pkl paths under dataset_path.

        Reads stems from samples.txt at the dataset root if present, else globs
        recursively (excluding `grasp_pred/`) and caches the list. Stems are
        paths relative to dataset_path (e.g. `scene/00000064` for a nested
        layout, or just `00000064` when flat). If `samples_filename` is provided
        that file must exist (subset request).
        """
        filename = samples_filename or "samples.txt"
        samples_file = dataset_path / filename
        if samples_filename is not None and not samples_file.exists():
            raise FileNotFoundError(f"Subset file not found: {samples_file}")
        if samples_file.exists():
            stems = samples_file.read_text().splitlines()
            files = [dataset_path / f"{stem}.pkl" for stem in stems if stem]
            logger.info(f"Loaded {len(files)} {split} files from {samples_file}")
            return files
        logger.info(
            f"Globbing {dataset_path} (no samples file, may take minutes on NFS)..."
        )
        files = GraspDataset.find_pkls(dataset_path)
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        stems = "\n".join(
            p.relative_to(dataset_path).with_suffix("").as_posix() for p in files
        )
        samples_file.write_text(stems + "\n")
        logger.info(f"Wrote {len(files)} {split} stems to {samples_file}")
        return files

    def _load_grasp_data(self, grasp_path: Path) -> Dict:
        with open(grasp_path, "rb") as f:
            return pickle.load(f)

    def _get_mano_params(self, grasp_data) -> torch.Tensor:
        """Extract 109D MANO state including the ten shape coefficients.

        Translation is metric [x, y, z], matching the model's PCL + 3D query
        point space.
        """
        grasp = grasp_data["grasp"]
        t = grasp["t"].flatten()
        R_6d = grasp["R_6d"].flatten()
        pose_6d = grasp["pose_6d"].flatten()
        parts = [t, R_6d, pose_6d]
        if self.d_mano == 109:
            # Converted DexYCB/HO3D samples keep both the canonical HUG
            # shape ("shape") and the source subject shape ("shape_gt").
            # 109D learning must use the latter; fall back for older pkls.
            shape_key = "shape_gt" if "shape_gt" in grasp else "shape"
            shape = np.asarray(grasp[shape_key], dtype=np.float32).flatten()
            if shape.size != 10:
                raise ValueError(f"expected 10 MANO shape coefficients, got {shape.size}")
            parts.append(shape)
        mano_params = np.concatenate(parts, axis=0).astype(np.float32)
        return torch.from_numpy(mano_params)

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes -> (H,W,3) uint8 RGB."""
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _decode_mask(mask_bytes: bytes) -> np.ndarray:
        """Decode PNG bytes -> (H,W) uint8 mask."""
        arr = np.frombuffer(mask_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _decode_depth_uint16(depth_bytes: bytes) -> np.ndarray:
        """Decode PNG bytes -> (H,W) uint16 depth (1mm units)."""
        arr = np.frombuffer(depth_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    def _depth_meters(self, depth_bytes: bytes) -> torch.Tensor:
        """Decode depth bytes to (H,W) float32 tensor in meters."""
        depth = self._decode_depth_uint16(depth_bytes).astype(np.float32)
        depth[depth >= 65535] = 0
        depth_m = np.nan_to_num(depth / 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
        # The PCL path rejects points at >=3 m. Apply the same physical bound
        # before query selection so invalid 7-13 m sensor codes cannot become
        # a crop center while the corresponding point cloud is empty.
        depth_m[(depth_m < 0.0) | (depth_m >= 3.0)] = 0.0
        return torch.from_numpy(depth_m)

    def _build_pcl(
        self,
        depth_m: torch.Tensor,
        rgb_np: np.ndarray,
        K: np.ndarray,
        point_xyz: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backproject depth + RGB into fixed-size (xyz, rgb_pcl) PCL tensors.

        When point_xyz and self.pcl_crop_radius are both set, restricts the
        point pool to a sphere of that radius around point_xyz before the
        random subsample, concentrating density on the grasp region.
        """
        crop_radius = self.pcl_crop_radius if point_xyz is not None else None
        return depth_to_pcl_tensors(
            depth_m,
            rgb_np,
            K,
            n_points=self.n_points_input,
            center=point_xyz,
            crop_radius=crop_radius,
            rng=rng,
        )

    @staticmethod
    def _query_sampling_debug(
        mask_np: np.ndarray,
        depth_np: np.ndarray,
        min_depth: float = 0.15,
        max_depth: float = 2.0,
        cluster_width: float = 0.12,
    ) -> Dict:
        """Build reliable query candidates from a mask and depth image.

        Stage-1 sampling keeps the existing converted masks unchanged, but
        rejects boundary/invalid-depth pixels and prefers pixels far from the
        mask boundary. The returned dictionary is also used by the dataloader
        visualizer to inspect the exact candidate region.
        """
        mask_bin = (np.asarray(mask_np) > 0.5).astype(np.uint8)
        depth = np.asarray(depth_np, dtype=np.float32)
        depth_valid = (
            np.isfinite(depth)
            & (depth >= float(min_depth))
            & (depth <= float(max_depth))
        )
        H, W = mask_bin.shape

        # Keep the largest connected mask component; tiny disconnected blobs
        # are conversion/compression artifacts, not hand pixels.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin, connectivity=8
        )
        component = np.zeros_like(mask_bin)
        if n_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            component = (labels == largest).astype(np.uint8)
        else:
            component = mask_bin.copy()

        area = int(component.sum())
        if area:
            ys, xs = np.where(component > 0)
            bbox_scale = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            erode_iters = int(np.clip(round(0.02 * bbox_scale), 1, 4))
            core = cv2.erode(
                component, np.ones((3, 3), np.uint8), iterations=erode_iters
            )
        else:
            erode_iters = 0
            core = np.zeros_like(component)

        # The converted mask is a projected MANO hull rather than a visible
        # segmentation. Keep the nearest well-supported metric-depth cluster
        # inside it. This favors the foreground hand/object surface over a
        # larger background region exposed by gaps in the convex hull.
        cluster_source = depth[(core > 0) & depth_valid]
        if cluster_source.size == 0:
            cluster_source = depth[(component > 0) & depth_valid]
        depth_cluster = np.zeros_like(depth_valid)
        cluster_center = float("nan")
        if cluster_source.size:
            bin_width = 0.05
            bins = np.arange(
                float(min_depth), float(max_depth) + bin_width, bin_width,
                dtype=np.float32,
            )
            hist, edges = np.histogram(cluster_source, bins=bins)
            support = max(5, int(np.ceil(float(hist.max()) * 0.10)))
            supported_bins = np.flatnonzero(hist >= support)
            peak = int(supported_bins[0]) if supported_bins.size else int(np.argmax(hist))
            in_peak = cluster_source[
                (cluster_source >= edges[peak])
                & (cluster_source <= edges[peak + 1])
            ]
            cluster_center = float(np.median(in_peak))
            depth_cluster = depth_valid & (
                np.abs(depth - cluster_center) <= float(cluster_width)
            )

        # A local valid-depth majority check avoids selecting isolated flying
        # pixels and single-pixel holes inside the dominant cluster.
        valid_count = cv2.boxFilter(
            depth_cluster.astype(np.float32),
            ddepth=-1,
            ksize=(5, 5),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        depth_median = cv2.medianBlur(depth, 5) if H >= 5 and W >= 5 else depth
        depth_tol = np.maximum(0.008, 0.03 * np.maximum(depth_median, 0.0))
        depth_consistent = (
            depth_cluster
            & (valid_count >= 13.0)
            & np.isfinite(depth_median)
            & (depth_median > 0)
            & (np.abs(depth - depth_median) <= depth_tol)
        )

        # Candidate priority never falls back outside the physical depth range
        # or dominant cluster.
        candidate_sets = (
            (core > 0) & depth_consistent,
            (core > 0) & depth_cluster,
            (component > 0) & depth_consistent,
            (component > 0) & depth_cluster,
        )
        candidate = np.zeros_like(component, dtype=bool)
        fallback_stage = "none"
        for stage, current in enumerate(candidate_sets):
            if current.any():
                candidate = current
                fallback_stage = (
                    "core_consistent",
                    "core_cluster",
                    "mask_consistent",
                    "mask_cluster",
                )[stage]
                break

        distance = (
            cv2.distanceTransform(component, cv2.DIST_L2, 5)
            if area
            else np.zeros_like(depth, dtype=np.float32)
        )
        return {
            "component": component,
            "core": core,
            "depth_consistent": depth_consistent.astype(np.uint8),
            "depth_cluster": depth_cluster.astype(np.uint8),
            "cluster_center": cluster_center,
            "candidate": candidate.astype(np.uint8),
            "distance": distance,
            "fallback_stage": fallback_stage,
            "erode_iters": erode_iters,
        }

    def _sample_point_from_mask(
        self, mask: torch.Tensor, depth_m: torch.Tensor
    ) -> torch.Tensor:
        """Sample a reliable hand pixel and return (u, v, d_meters).

        Pixels near mask boundaries, invalid depth, and locally inconsistent
        depth are rejected. The remaining pixels are sampled 70% with a
        distance-to-boundary weighting and 30% uniformly for coverage.
        """
        mask_np = (mask.squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8)
        depth_np = (
            depth_m.detach().cpu().numpy()
            if isinstance(depth_m, torch.Tensor)
            else np.asarray(depth_m)
        )
        debug = self._query_sampling_debug(
            mask_np,
            depth_np,
            min_depth=self.query_min_depth,
            max_depth=self.query_max_depth,
            cluster_width=self.query_depth_cluster_width,
        )
        candidate = debug["candidate"] > 0

        if candidate.any():
            ys, xs = np.where(candidate)
            if self.split != "train":
                # Validation/test must use the same condition at every
                # checkpoint. Prefer the deepest interior candidate.
                idx = int(np.argmax(debug["distance"][ys, xs]))
            elif np.random.random() < 0.70:
                weights = debug["distance"][ys, xs].astype(np.float64)
                weights = np.maximum(weights, 1e-6)
                weights /= weights.sum()
                idx = int(np.random.choice(len(ys), p=weights))
            else:
                idx = int(np.random.randint(len(ys)))
            v_pix, u_pix = int(ys[idx]), int(xs[idx])
            d = float(depth_np[v_pix, u_pix])
            return torch.tensor([float(u_pix), float(v_pix), d], dtype=torch.float32)

        # Extremely rare samples with an empty mask or no valid depth are kept
        # finite without inventing a background depth. Clean training lists
        # should remove these samples; this branch is only a last-resort guard.
        if (mask_np > 0).any():
            ys, xs = np.where(mask_np > 0)
            center = np.array([float(np.median(xs)), float(np.median(ys))])
            k = int(np.argmin((xs - center[0]) ** 2 + (ys - center[1]) ** 2))
            u_pix, v_pix = int(xs[k]), int(ys[k])
        else:
            height, width = mask_np.shape
            u_pix, v_pix = width // 2, height // 2
        return torch.tensor([float(u_pix), float(v_pix), 0.0], dtype=torch.float32)

    def _robust_depth_at_pixel(
        self, depth_np: np.ndarray, u: float, v: float
    ) -> float:
        """Median valid depth near a stored query, without whole-image fallback."""
        depth = np.asarray(depth_np, dtype=np.float32)
        height, width = depth.shape
        ui = int(np.clip(round(u), 0, width - 1))
        vi = int(np.clip(round(v), 0, height - 1))
        physically_valid = (
            np.isfinite(depth)
            & (depth >= self.query_min_depth)
            & (depth <= self.query_max_depth)
        )
        for radius in (7, 15, 31):
            window = depth[
                max(0, vi - radius) : min(height, vi + radius + 1),
                max(0, ui - radius) : min(width, ui + radius + 1),
            ]
            valid = physically_valid[
                max(0, vi - radius) : min(height, vi + radius + 1),
                max(0, ui - radius) : min(width, ui + radius + 1),
            ]
            values = window[valid]
            if values.size:
                return float(np.median(values))
        return 0.0

    def get_original_for_viz(self, idx: int) -> Dict[str, np.ndarray]:
        """Load 224-res data from pkl for 3D viz; no external files needed."""
        grasp_path = self.grasp_files[idx]
        grasp_data = self._load_grasp_data(grasp_path)

        rgb_small = self._decode_image(grasp_data["image"])
        depth_image = self._decode_depth_uint16(grasp_data["depth"])
        camera_K_small = grasp_data["camera"]["K"]

        key = grasp_path.relative_to(self.dataset_path).with_suffix("").as_posix()

        return {
            "rgb_small": rgb_small,
            "depth_image": depth_image,
            "camera_K_small": camera_K_small,
            "stem": key,
        }

    def get_inference_data(self, stem: str) -> Dict:
        """Load minimal data needed for inference: rgb, depth, camera, mesh_faces."""
        pkl_path = self.dataset_path / f"{stem}.pkl"
        grasp_data = self._load_grasp_data(pkl_path)
        camera = grasp_data["camera"]
        K = camera["K"] if isinstance(camera, dict) else camera.K
        width = camera["width"] if isinstance(camera, dict) else camera.width
        height = camera["height"] if isinstance(camera, dict) else camera.height

        grasp = grasp_data.get("grasp")
        mesh_faces = grasp.get("mesh_faces") if grasp else None
        if mesh_faces is None:
            mesh_faces = np.load(MANO_RIGHT_MESH_FACES_FILE)

        rgb_np = self._decode_image(grasp_data["image"])
        rgb_original_path = self.dataset_path / "image_original" / f"{stem}.jpg"
        rgb_original = (
            np.array(Image.open(rgb_original_path).convert("RGB"))
            if rgb_original_path.exists()
            else rgb_np
        )

        depth_image = self._decode_depth_uint16(grasp_data["depth"])
        shape = (
            (grasp.get("shape_gt", grasp["shape"]) if grasp else
             np.load(MANO_RIGHT_SHAPE_FILE).reshape(1, 10))
        )
        mano_shape = torch.from_numpy(np.asarray(shape).flatten()).float()

        out = {
            "camera_K": K,
            "width": width,
            "height": height,
            "mesh_faces": mesh_faces,
            "rgb_original": rgb_original,
            "depth_image": depth_image,
            "mano_shape": mano_shape,
        }
        if self.use_rgb:
            out["rgb"] = self.rgb_transform(Image.fromarray(rgb_np))
        if self.use_depth:
            depth_m = self._depth_meters(grasp_data["depth"])
            K_224 = grasp_data["camera"]["K"]
            xyz, pcl_rgb = self._build_pcl(depth_m, rgb_np, K_224)
            out["pcl_xyz"] = xyz
            out["pcl_rgb"] = pcl_rgb
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        grasp_path = self.grasp_files[idx]
        grasp_data = self._load_grasp_data(grasp_path)
        # Root-relative stem so nested layouts round-trip back to the pkl path
        # (matches app/inference resolution via dataset_path / f"{stem}.pkl").
        stem = grasp_path.relative_to(self.dataset_path).with_suffix("").as_posix()

        # Depth stays in the full-frame coordinate system (after any shared
        # affine augmentation). Only the RGB tensor sent to DINO is cropped.
        depth_m = self._depth_meters(grasp_data["depth"])
        depth_for_pcl = depth_m
        K_np = np.asarray(grasp_data["camera"]["K"], dtype=np.float32)
        rgb_original = self._decode_image(grasp_data["image"])
        rgb_np = rgb_original
        rgb_K_np = K_np.copy()

        crop_bbox = np.array(
            [
                0.0,
                0.0,
                float(rgb_original.shape[1] - 1),
                float(rgb_original.shape[0] - 1),
            ],
            dtype=np.float32,
        )
        crop_valid = True
        crop_source = "disabled"
        crop_affine = np.eye(3, dtype=np.float32)[:2]
        keypoints_2d = np.zeros((21, 2), dtype=np.float32)
        keypoints_valid = np.zeros(21, dtype=np.bool_)
        depth_crop_center = np.zeros(2, dtype=np.float32)
        # A negative radius explicitly means that no 2D keypoint crop was
        # applied; that path uses HUG's query-centred metric sphere instead.
        depth_crop_radius = -1.0
        has_hand_keypoints = False

        if self.hand_crop_enabled:
            crop_bbox, crop_valid, crop_source = self._select_hand_crop_bbox(
                grasp_data, rgb_original
            )
            rgb_np, rgb_K_np, crop_affine = self._crop_rgb(
                rgb_original, K_np, crop_bbox
            )
            (
                keypoints_2d,
                keypoints_valid,
                keypoints_original,
                depth_keypoints_valid,
            ) = self._hand_keypoints(
                grasp_data, rgb_np, rgb_original, crop_affine
            )
            has_hand_keypoints = int(depth_keypoints_valid.sum()) >= 2
            if has_hand_keypoints:
                depth_for_pcl, depth_crop_center, depth_crop_radius = (
                    self._crop_original_depth_by_keypoint_radius(
                        depth_m,
                        keypoints_original,
                        depth_keypoints_valid,
                    )
                )
                anchor_uv = self._palm_anchor(
                    keypoints_original, depth_keypoints_valid
                )
                d = self._robust_depth_at_pixel(
                    depth_m.numpy(), float(anchor_uv[0]), float(anchor_uv[1])
                )
                point_uv = torch.tensor(
                    [float(anchor_uv[0]), float(anchor_uv[1]), d], dtype=torch.float32
                )
            else:
                # No skeleton means this may be an object-only generation
                # sample (or a detector miss). Restore the full RGB view and
                # retain the original HUG query + metric point-cloud crop.
                rgb_np = rgb_original
                rgb_K_np = K_np.copy()
                crop_bbox = np.array(
                    [
                        0.0,
                        0.0,
                        float(rgb_original.shape[1] - 1),
                        float(rgb_original.shape[0] - 1),
                    ],
                    dtype=np.float32,
                )
                crop_affine = np.eye(3, dtype=np.float32)[:2]
                crop_valid = False
                crop_source = f"{crop_source}_no_keypoints_full"
                stored_uv = grasp_data.get("condition_point")
                if stored_uv is not None:
                    u, v = float(stored_uv[0]), float(stored_uv[1])
                    d = self._robust_depth_at_pixel(depth_m.numpy(), u, v)
                    point_uv = torch.tensor([u, v, d], dtype=torch.float32)
                else:
                    mask_np = self._decode_mask(grasp_data["object_mask"])
                    mask_tensor = self.mask_transform(Image.fromarray(mask_np))
                    point_uv = self._sample_point_from_mask(mask_tensor, depth_m)
                depth_crop_center = point_uv[:2].numpy().copy()
        else:
            # Legacy HUG query path retained for old configs/checkpoints and
            # object-only grasp generation samples.
            stored_uv = grasp_data.get("condition_point")
            if stored_uv is not None:
                u, v = float(stored_uv[0]), float(stored_uv[1])
                d = self._robust_depth_at_pixel(depth_m.numpy(), u, v)
                point_uv = torch.tensor([u, v, d], dtype=torch.float32)
            else:
                mask_np = self._decode_mask(grasp_data["object_mask"])
                mask_tensor = self.mask_transform(Image.fromarray(mask_np))
                point_uv = self._sample_point_from_mask(mask_tensor, depth_m)

        camera_K = torch.from_numpy(K_np).float()
        rgb_camera_K = torch.from_numpy(rgb_K_np).float()

        out = {
            "point_uv": point_uv,
            "camera_K": camera_K,
            "rgb_camera_K": rgb_camera_K,
            "stem": stem,
            "hand_keypoints_2d": torch.from_numpy(keypoints_2d).float(),
            "hand_keypoints_valid": torch.from_numpy(keypoints_valid),
            "hand_crop_bbox": torch.from_numpy(crop_bbox).float(),
            "hand_crop_valid": torch.tensor(bool(crop_valid)),
            "hand_crop_source": crop_source,
            "hand_keypoint_source": (
                self.keypoint_source if self.hand_crop_enabled else "disabled"
            ),
            "depth_crop_center_uv": torch.from_numpy(depth_crop_center).float(),
            "depth_crop_radius_px": torch.tensor(depth_crop_radius).float(),
            "has_hand_keypoints": torch.tensor(bool(has_hand_keypoints)),
            "pcl_crop_mode": (
                "keypoint_radius" if has_hand_keypoints else "query_sphere"
            ),
            "query_valid": torch.tensor(
                self.query_min_depth <= float(point_uv[2]) <= self.query_max_depth
            ),
        }
        # Eval pkls carry no grasp label; GT fields are train-only
        grasp = grasp_data.get("grasp")
        if grasp is not None:
            out["mano_params"] = self._get_mano_params(grasp_data)
            shape_key = "shape_gt" if "shape_gt" in grasp else "shape"
            out["mano_shape"] = torch.from_numpy(grasp[shape_key].flatten()).float()
            out["landmarks_3d"] = torch.from_numpy(grasp["landmarks_3d"]).float()
            out["landmarks_2d"] = torch.from_numpy(
                np.asarray(grasp["landmarks_2d"], dtype=np.float32)
            ).float()
        elif "joints_gt" in grasp_data:
            # HO3D_v3 evaluation split: GT is joints/verts only, no MANO params.
            # Reorder joints from the official raw order to our standard order
            # (see HO3D_RAW_TO_STD above); verts are MANO-template-ordered
            # already and pass through unchanged.
            joints_gt = np.asarray(grasp_data["joints_gt"], dtype=np.float32)[HO3D_RAW_TO_STD]
            out["joints_gt"] = torch.from_numpy(joints_gt)
            out["verts_gt"] = torch.from_numpy(
                np.asarray(grasp_data["verts_gt"], dtype=np.float32)
            )
        if self.use_rgb:
            out["rgb"] = self.rgb_transform(Image.fromarray(rgb_np))
        if self.use_depth:
            # A detected skeleton already supplies an adaptive 2D crop, so do
            # not stack a fixed metric sphere on top. Without a skeleton,
            # fall back to HUG's query-centred sphere (normally 0.30 m).
            point_xyz = None
            if bool(out["query_valid"]) and not has_hand_keypoints:
                point_xyz = pixel_to_xyz(
                    float(point_uv[0]), float(point_uv[1]), float(point_uv[2]), K_np
                )
            pcl_rng = None
            if self.split != "train":
                pcl_rng = np.random.default_rng(zlib.crc32(stem.encode("utf-8")))
            xyz, pcl_rgb = self._build_pcl(
                depth_for_pcl,
                rgb_original,
                K_np,
                point_xyz=point_xyz,
                rng=pcl_rng,
            )
            out["pcl_xyz"] = xyz
            out["pcl_rgb"] = pcl_rgb
        return out

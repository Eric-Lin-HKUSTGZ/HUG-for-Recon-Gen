"""DDP training for HUG (GraspFlowModel), configured by a YAML file.

Launch (multi-GPU):
    torchrun --nproc_per_node=4 -m src.train --config configs/train_hug.yaml

Smoke test (short run on a subset):
    torchrun --nproc_per_node=4 -m src.train --config configs/train_hug.yaml \
        --max-steps 30 --max-train-samples 20000

Loss (HUG paper Eq. 1 plus mesh supervision):
    L = λv Lflow + E[(1 - t) (λ3D Ljoints + λmesh Lmesh)]
  Lflow: velocity-prediction MSE over the complete normalized state.
  Ljoints: camera-frame L1 on the 21 MANO landmarks decoded from x0_hat.
  Lmesh: camera-frame L1 on all 778 MANO vertices decoded from x0_hat.
  The reconstruction model keeps its 109D state, so the flow MSE also covers
  the ten learned MANO shape dimensions added to HUG's original 99D state.

LR schedule: linear warmup (warmup_steps) -> cosine decay to lr * lr_min_ratio.

Checkpoints embed cfg + norm_stats + EMA weights in the exact layout that
`src/inference.py:load_model` consumes ({model, ema, cfg, norm_stats, ...}),
so a training output dir can be pointed at inference/app directly.
"""

import atexit
import faulthandler
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tyro
from omegaconf import OmegaConf
from rich.console import Console
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard is optional; JSONL remains the fallback.
    SummaryWriter = None

from .dataloader.augmented_grasp_dataset import AugmentedGraspDataset
from .dataloader.grasp_dataset import GraspDataset
from .metrics import joint_mesh_errors
from .models.grasp_model import GraspFlowModel
from .utils.data_keys import NORM_STATS_FILE

logger = logging.getLogger("hug.train")
console = Console()

LOG_SCHEMA_VERSION = "train-log-v2"

# Dataset normalization defines the coordinate system used by flow matching.
# A finetune run must keep the stats supplied by its config instead of silently
# restoring the source dataset's stats from a pretrained state dict. Resume is
# intentionally unaffected: it restores the complete checkpoint state below.
_PRETRAINED_NORM_BUFFER_KEYS = frozenset(
    f"flow.denoise_fn.{name}"
    for name in (
        "trans_mean",
        "trans_std",
        "wrist_mean",
        "wrist_std",
        "finger_mean",
        "finger_std",
        "shape_mean",
        "shape_std",
    )
)


class ContextFormatter(logging.Formatter):
    """Add DDP/process context to every durable text-log record."""

    def format(self, record):
        base = super().format(record)
        return (
            f"[rank={getattr(record, 'rank', os.environ.get('RANK', '0'))} "
            f"local_rank={getattr(record, 'local_rank', os.environ.get('LOCAL_RANK', '0'))} "
            f"pid={os.getpid()}] {base}"
        )


class JsonlWriter:
    """Flush-on-write structured event writer; only rank 0 owns metrics JSONL."""

    def __init__(self, path: Path, run_id: str, rank: int, world_size: int):
        self.path = path
        self.run_id = run_id
        self.rank = rank
        self.world_size = world_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict | None = None, step: int | None = None):
        record = {
            "schema_version": LOG_SCHEMA_VERSION,
            "event": event,
            "run_id": self.run_id,
            "time": datetime.now(timezone.utc).isoformat(),
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if step is not None:
            record["step"] = int(step)
        if payload:
            record.update(payload)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if os.environ.get("HUG_LOG_FSYNC", "0") == "1":
                os.fsync(f.fileno())


_TB_TRAIN_TAGS = {
    "loss": "loss/total",
    "loss_flow_weighted": "loss_weighted/flow",
    "loss_3d_weighted": "loss_weighted/3d_landmarks",
    "loss_mesh_3d_weighted": "loss_weighted/3d_mesh",
    "lv": "loss_raw/flow",
    "l3d": "loss_raw/3d_landmarks",
    "l3d_time_weighted": "loss_raw/3d_landmarks_time_weighted",
    "lmesh": "loss_raw/3d_mesh",
    "lmesh_time_weighted": "loss_raw/3d_mesh_time_weighted",
    "train_mpjpe_mm": "error_mm/train_mpjpe",
    "train_mpvpe_mm": "error_mm/train_mpvpe",
    "time_weight_mean": "schedule/one_minus_t_mean",
    "lr": "optimization/learning_rate",
    "grad_norm": "optimization/gradient_norm",
    "samples_per_s": "performance/steps_per_second",
}


def _create_tensorboard_writer(train_cfg, out_dir: Path, start_step: int, cfg=None):
    """Create the rank-0 TensorBoard writer with resume-safe step purging."""
    tb_cfg = train_cfg.get("tensorboard", {})
    if not bool(tb_cfg.get("enabled", True)):
        return None, None
    if SummaryWriter is None:
        logger.warning(
            "TensorBoard is enabled but the tensorboard package is unavailable; "
            "continuing with JSONL logging"
        )
        return None, None

    configured_dir = tb_cfg.get("log_dir", "tensorboard")
    log_dir = Path(str(configured_dir))
    if not log_dir.is_absolute():
        log_dir = out_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    purge_step = start_step + 1 if start_step > 0 else None
    tb_writer = SummaryWriter(
        log_dir=str(log_dir),
        purge_step=purge_step,
        max_queue=max(int(tb_cfg.get("max_queue", 10)), 1),
        flush_secs=max(int(tb_cfg.get("flush_secs", 10)), 1),
    )
    tb_writer.add_custom_scalars(
        {
            "Loss": {
                "weighted_components": [
                    "Multiline",
                    [
                        "loss_weighted/flow",
                        "loss_weighted/3d_landmarks",
                        "loss_weighted/3d_mesh",
                    ],
                ],
                "raw_components": [
                    "Multiline",
                    [
                        "loss_raw/flow",
                        "loss_raw/3d_landmarks",
                        "loss_raw/3d_landmarks_time_weighted",
                        "loss_raw/3d_mesh",
                        "loss_raw/3d_mesh_time_weighted",
                    ],
                ],
            },
            "Train error": {
                "Camera-frame geometry (mm)": [
                    "Multiline",
                    ["error_mm/train_mpjpe", "error_mm/train_mpvpe"],
                ]
            },
        }
    )
    if cfg is not None:
        tb_writer.add_text(
            "run/config",
            "```yaml\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n```",
            start_step,
        )
    tb_writer.add_text(
        "run/status",
        f"TensorBoard initialized at training step {start_step}.",
        start_step,
    )
    tb_writer.flush()
    return tb_writer, log_dir


def _close_tensorboard_writer(tb_writer):
    """Flush TensorBoard on normal completion and uncaught Python exceptions."""
    if tb_writer is None:
        return
    try:
        tb_writer.flush()
        tb_writer.close()
    except Exception:
        logger.warning("TensorBoard close failed", exc_info=True)


def _write_tensorboard_scalars(tb_writer, event: str, payload: dict, step: int):
    """Write stable, explicitly grouped TensorBoard tags."""
    if tb_writer is None:
        return
    try:
        if event == "train":
            for key, tag in _TB_TRAIN_TAGS.items():
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(tag, value, step)
        elif event == "validation":
            for dataset, metrics in payload.get("datasets", {}).items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tb_writer.add_scalar(f"validation/{dataset}/{key}_mm", value, step)
            score = payload.get("score")
            if isinstance(score, (int, float)):
                tb_writer.add_scalar("validation/selection_score_mm", score, step)
        tb_writer.flush()
    except Exception:
        # Monitoring must never change training correctness or availability.
        logger.warning("TensorBoard write failed; continuing with JSONL logging", exc_info=True)


def _resolve_log_path(train_cfg, out_dir: Path) -> Path:
    log_file = train_cfg.get("log_file")
    return (
        Path(log_file)
        if log_file and Path(log_file).is_absolute()
        else out_dir / (log_file or "train_log.jsonl")
    )


def configure_logging(out_dir: Path, log_path: Path, rank: int, local_rank: int, world_size: int):
    """Install durable rank-aware logging before data/model initialization."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    text_path = log_dir / f"rank-{rank}.log"
    error_path = log_dir / f"rank-{rank}.error.log"
    fmt = ContextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for path, level in ((text_path, logging.INFO), (error_path, logging.ERROR)):
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    logging.captureWarnings(True)
    faulthandler.enable()
    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or uuid.uuid4().hex[:12]
    writer = JsonlWriter(log_path, run_id, rank, world_size) if is_main(rank) else None
    context = {"rank": rank, "local_rank": local_rank, "world_size": world_size}
    logger.info(
        "logging initialized: output=%s metrics=%s text=%s errors=%s host=%s",
        out_dir, log_path, text_path, error_path, socket.gethostname(), extra=context,
    )
    return writer, context


def _safe_event(writer, logger_obj, event, payload=None, step=None):
    """Write an event without hiding the original failure if logging breaks."""
    if writer is None:
        return
    try:
        writer.write(event, payload, step=step)
    except Exception:
        logger_obj.exception("failed to write structured event %s", event)


def _phase_log(context, phase, message, *args):
    logger.info("phase=%s %s", phase, message % args if args else message, extra=context)


def install_exception_hooks(context, state=None):
    """Persist uncaught main/thread exceptions with full tracebacks."""
    state = state if state is not None else {}

    def handle(exc_type, exc_value, exc_tb):
        context_now = {
            **context,
            "step": state.get("step"),
            "phase": state.get("phase"),
        }
        if issubclass(exc_type, KeyboardInterrupt):
            logger.error("interrupted", exc_info=(exc_type, exc_value, exc_tb), extra=context_now)
        else:
            logger.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_tb), extra=context_now)
        logging.shutdown()
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = handle
    if hasattr(threading, "excepthook"):
        def thread_hook(args):
            context_now = {
                **context,
                "step": state.get("step"),
                "phase": state.get("phase"),
            }
            logger.critical("uncaught thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback), extra=context_now)
            logging.shutdown()
        threading.excepthook = thread_hook



# --------------------------------------------------------------------------
# DDP helpers
# --------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    """Init process group from torchrun env vars. Returns (rank, local_rank, world_size, device)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        # 30min watchdog: val sampling + vepfs checkpoint writes can exceed the
        # default 10min NCCL timeout
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _dataset_class(data_cfg):
    """Select the loader without changing the original GraspDataset API."""
    aug_cfg = data_cfg.get("augmentation", {})
    return AugmentedGraspDataset if bool(aug_cfg.get("enabled", False)) else GraspDataset


def _make_dataset(
    data_cfg,
    model_cfg,
    dataset_path,
    split,
    samples_filename=None,
    indices=None,
):
    """Construct one dataset with augmentation scoped to training."""
    dataset_cls = _dataset_class(data_cfg)
    kwargs = dict(
        dataset_path=str(dataset_path),
        split=split,
        indices=indices,
        image_size=model_cfg.get("image_size", 224),
        n_points_input=data_cfg.n_points_input,
        pcl_crop_radius=model_cfg.get("pcl_crop_radius", 0.3),
        d_mano=int(model_cfg.get("d_mano", 99)),
        query_min_depth=float(data_cfg.get("query_min_depth", 0.15)),
        query_max_depth=float(data_cfg.get("query_max_depth", 2.0)),
        query_depth_cluster_width=float(
            data_cfg.get("query_depth_cluster_width", 0.12)
        ),
        use_rgb=model_cfg.get("use_rgb", True),
        use_depth=model_cfg.get("use_depth", True),
        samples_filename=(
            str(samples_filename) if samples_filename is not None else None
        ),
    )
    if dataset_cls is AugmentedGraspDataset:
        kwargs["augmentation"] = data_cfg.get("augmentation", {})
    return dataset_cls(**kwargs)

def build_datasets(cfg):
    """Train dataset + val dataset.

    Two config styles:

    1. Multi-dataset (preferred): `trainer.data.datasets` is a list of
       {path, train_samples, val_samples} — explicit stem-list files per
       dataset (absolute paths OK). Train/val are ConcatDatasets over entries.
    2. Legacy single-dataset: `trainer.data.dataset_path` + recording-level
       `val_split_file` (built by scripts/make_val_split.py), falling back to
       a deterministic random frame-level split.
    """
    data_cfg = cfg.trainer.data
    if data_cfg.get("datasets"):
        train_parts, val_parts = [], []
        for entry in data_cfg.datasets:
            train_parts.append(
                _make_dataset(
                    data_cfg,
                    cfg.trainer.model,
                    entry.path,
                    split="train",
                    samples_filename=entry.train_samples,
                )
            )
            val_parts.append(
                _make_dataset(
                    data_cfg,
                    cfg.trainer.model,
                    entry.path,
                    split="val",
                    samples_filename=entry.val_samples,
                )
            )
            logger.info(
                f"{entry.path}: train={len(train_parts[-1])} val={len(val_parts[-1])}"
            )
        from torch.utils.data import ConcatDataset

        train_ds = (
            train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        )
        val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
        return train_ds, val_ds

    dataset_path = str(data_cfg.dataset_path)
    full = _make_dataset(
        data_cfg,
        cfg.trainer.model,
        dataset_path,
        split="all",
        samples_filename=data_cfg.get("samples_filename"),
    )
    n = len(full)
    root = Path(dataset_path)

    split_file = root / data_cfg.get("val_split_file", "split_val.txt")
    if split_file.exists():
        val_stems = set(split_file.read_text().splitlines())
        val_idx, train_idx = [], []
        for i, p in enumerate(full.grasp_files):
            stem = p.relative_to(root).with_suffix("").as_posix()
            (val_idx if stem in val_stems else train_idx).append(i)
        logger.info(
            f"Recording-level split from {split_file.name}: "
            f"{len(val_idx)} val / {len(train_idx)} train frames"
        )
    else:
        logger.warning(
            f"{split_file} not found — falling back to random frame-level split "
            f"(leaks across frames of one grasp; run scripts/make_val_split.py)"
        )
        rng = np.random.default_rng(42)
        val_count = min(int(data_cfg.get("val_count", 512)), n // 10)
        val_idx = sorted(rng.choice(n, size=val_count, replace=False).tolist())
        val_set = set(val_idx)
        train_idx = [i for i in range(n) if i not in val_set]

    train_ds = _make_dataset(
        data_cfg, cfg.trainer.model, dataset_path, split="train", indices=train_idx
    )
    val_ds = _make_dataset(
        data_cfg, cfg.trainer.model, dataset_path, split="val", indices=val_idx
    )
    return train_ds, val_ds


def build_loaders(cfg, train_ds, val_ds, rank, world_size, max_train_samples=None):
    train_cfg = cfg.trainer.train
    if max_train_samples is not None:
        if hasattr(train_ds, "grasp_files"):
            train_ds.grasp_files = train_ds.grasp_files[:max_train_samples]
        else:  # ConcatDataset: cap per part proportionally
            from torch.utils.data import Subset

            train_ds = Subset(
                train_ds,
                list(range(min(max_train_samples, len(train_ds)))),
            )
    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    num_workers = int(cfg.trainer.data.get("num_workers", 0))
    train_loader_kwargs = dict(
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # PyTorch rejects persistent_workers/prefetch_factor when num_workers=0;
    # omitting them keeps low-resource smoke tests and single-process runs
    # valid while preserving the configured prefetch behavior for workers.
    if num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(cfg.trainer.data.get("prefetch_factor", 2)),
        )
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    # Val is evaluated on rank 0 only (see run_val); plain loaders suffice.
    val_cfg = cfg.trainer.get("val")

    if val_cfg is not None and val_cfg.get("datasets"):
        # New style: per-dataset val loaders (mixed GT schemas allowed - e.g.
        # DexYCB s0_val with MANO GT + HO3D_v3 eval with joints/verts GT).
        val_loaders = []
        parts = []
        for entry in val_cfg.datasets:
            ds = _make_dataset(
                cfg.trainer.data,
                cfg.trainer.model,
                entry.path,
                split="val",
                samples_filename=entry.samples,
            )
            parts.append(ds)
        total = sum(len(p) for p in parts)
        max_total = val_cfg.get("max_samples")
        if max_total is not None and total > int(max_total):
            # 各数据集按占比等距采样，确定性、跨 checkpoint 可比
            for ds in parts:
                k = max(1, round(len(ds) * int(max_total) / total))
                idx = sorted({round(i * len(ds) / k) for i in range(k)})
                ds.grasp_files = [ds.grasp_files[i] for i in idx]
            logger.info(
                f"val subsample -> "
                + ", ".join(f"{e.name}={len(p)}" for e, p in zip(val_cfg.datasets, parts))
            )
        for entry, ds in zip(val_cfg.datasets, parts):
            loader = DataLoader(
                ds,
                batch_size=train_cfg.batch_size,
                # 多卡分片并行验证：每 rank 评估自己的 shard，run_val 末尾
                # all_reduce 聚合。注意 DistributedSampler 会补齐到整除
                # world_size（重复末几条样本），对指标影响可忽略。
                sampler=(
                    DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
                    if world_size > 1
                    else None
                ),
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            val_loaders.append((str(entry.name), loader))
            logger.info(f"val dataset {entry.name}: {len(ds)} samples")
        return train_loader, val_loaders, sampler

    # Legacy: single val dataset (ConcatDataset style)
    max_val = cfg.trainer.data.get("max_val_samples")
    if max_val is not None and len(val_ds) > int(max_val):
        from torch.utils.data import Subset

        # 等距采样拼接后的整个 val 集（而非取前 N 条，否则只覆盖列表首个
        # 数据集）：各数据集按占比混合，且采样确定性、跨 checkpoint 可比
        n = len(val_ds)
        k = int(max_val)
        val_ds = Subset(
            val_ds, sorted({round(i * n / k) for i in range(k)})
        )
        from torch.utils.data import ConcatDataset

        if isinstance(val_ds.dataset, ConcatDataset):
            parts = val_ds.dataset.datasets
            bounds = [sum(map(len, parts[:j])) for j in range(1, len(parts) + 1)]
            origin = [
                sum(1 for i in val_ds.indices if lo <= i < hi)
                for lo, hi in zip([0] + bounds, bounds)
            ]
            logger.info(f"val subsample {len(val_ds)}: per-dataset {origin}")
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        sampler=(
            DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            if world_size > 1
            else None
        ),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, [("val", val_loader)], sampler


def infinite_loader(loader, sampler):
    """Yield batches forever, re-shuffling at every pass (step-based training)."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# --------------------------------------------------------------------------
# Loss (paper Eq. 1)
# --------------------------------------------------------------------------

def compute_loss(
    preds,
    targets,
    time_weight,
    lambda_v=1.0,
    lambda_3d=20.0,
    lambda_mesh_3d=10.0,
):
    """Compute HUG Eq. 1 with an additional MANO mesh constraint.

    ``Lv`` is one MSE over the complete normalized velocity tensor. ``L3D``
    is a camera-frame L1 loss over the 21 MANO landmarks decoded from the
    estimated clean state. ``Lmesh`` applies the same camera-frame L1 loss to
    all MANO vertices. Both decoded geometry terms are weighted per sample by
    ``1 - t``; ``time_weight`` is produced by :class:`GraspFlowModel` with
    exactly that value.

    The original HUG state is 99D. This reconstruction model intentionally
    remains 109D, so its ten MANO shape dimensions participate in ``Lv`` and
    influence ``L3D`` through the differentiable MANO decode.
    """
    velocity_pred = preds["params_norm"].float()
    velocity_target = targets["params_norm"].float()
    lv = F.mse_loss(velocity_pred, velocity_target)

    pred_joints = preds["landmarks_3d"].float()
    target_joints = targets["landmarks_3d"].float()
    l3d_per_sample = F.l1_loss(
        pred_joints,
        target_joints,
        reduction="none",
    ).mean(dim=(1, 2))
    pred_vertices = preds["vertices"].float()
    target_vertices = targets["vertices"].float()
    lmesh_per_sample = F.l1_loss(
        pred_vertices,
        target_vertices,
        reduction="none",
    ).mean(dim=(1, 2))

    time_weight = time_weight.float().reshape(-1)
    if time_weight.shape[0] != l3d_per_sample.shape[0]:
        raise ValueError(
            "time_weight batch size must match the decoded MANO landmarks"
        )
    l3d = l3d_per_sample.mean()
    l3d_time_weighted = (time_weight * l3d_per_sample).mean()
    lmesh = lmesh_per_sample.mean()
    lmesh_time_weighted = (time_weight * lmesh_per_sample).mean()

    loss_flow_weighted = float(lambda_v) * lv
    loss_3d_weighted = float(lambda_3d) * l3d_time_weighted
    loss_mesh_3d_weighted = float(lambda_mesh_3d) * lmesh_time_weighted
    loss = loss_flow_weighted + loss_3d_weighted + loss_mesh_3d_weighted

    comps = {
        "loss": loss.item(),
        "lv": lv.item(),
        "l3d": l3d.item(),
        "l3d_time_weighted": l3d_time_weighted.item(),
        "lmesh": lmesh.item(),
        "lmesh_time_weighted": lmesh_time_weighted.item(),
        "loss_flow_weighted": loss_flow_weighted.item(),
        "loss_3d_weighted": loss_3d_weighted.item(),
        "loss_mesh_3d_weighted": loss_mesh_3d_weighted.item(),
        "time_weight_mean": time_weight.mean().item(),
        "train_mpjpe_mm": torch.linalg.vector_norm(
            pred_joints - target_joints, dim=-1
        ).mean().item()
        * 1000.0,
        "train_mpvpe_mm": torch.linalg.vector_norm(
            pred_vertices - target_vertices, dim=-1
        ).mean().item()
        * 1000.0,
    }

    return loss, comps


# --------------------------------------------------------------------------
# Validation (rank 0, on the unwrapped module — no DDP collectives involved)
# --------------------------------------------------------------------------

METRIC_KEYS = ("mpjpe", "pa_mpjpe", "mpvpe", "pa_mpvpe")

# Version tag stored in checkpoints; a resume from a checkpoint saved with a
# different validation score resets best_val so model_best.pt comparisons use
# the same semantics. The best checkpoint minimizes mean PA-MPJPE/PA-MPVPE.
VAL_METRIC = "sampling-pa-mean-v3"


def run_val(raw_model, val_loaders, device, bf16, rank, world_size):
    """Sampling-based validation: real inference quality, no loss.

    val_loaders: list of (name, DataLoader). With world_size > 1 each rank
    evaluates its DistributedSampler shard and metric sums are all_reduced.
    Each batch routes by its GT schema: `mano_params` (DexYCB) ->
    build_loss_dicts; `joints_gt/verts_gt` (HO3D_v3 evaluation split, no MANO
    GT) -> mano_forward. Returns {name: {metric: mean_mm}} per dataset.
    """
    raw_model.eval()
    results = {}
    for name, loader in val_loaders:
        sums = {k: 0.0 for k in METRIC_KEYS}
        n = 0
        with torch.no_grad():
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    samples = raw_model.sample(
                        point_uv=batch["point_uv"].to(device),
                        camera_K=batch["camera_K"].to(device),
                        rgb=batch["rgb"].to(device) if "rgb" in batch else None,
                        pcl_xyz=batch["pcl_xyz"].to(device) if "pcl_xyz" in batch else None,
                        pcl_rgb=batch["pcl_rgb"].to(device) if "pcl_rgb" in batch else None,
                    )
                    if "mano_params" in batch:
                        preds, targets = raw_model.build_loss_dicts(
                            samples, batch["mano_params"].to(device)
                        )
                        errs = joint_mesh_errors(
                            preds["landmarks_3d"].float(),
                            targets["landmarks_3d"].float(),
                            preds["vertices"].float(),
                            targets["vertices"].float(),
                        )
                    else:
                        pred_out = raw_model.mano_forward(samples)
                        errs = joint_mesh_errors(
                            pred_out["landmarks_3d"].float(),
                            batch["joints_gt"].to(device).float(),
                            pred_out["vertices"].float(),
                            batch["verts_gt"].to(device).float(),
                        )
                for k in METRIC_KEYS:
                    sums[k] += errs[k].float().sum().item()
                n += errs["mpjpe"].shape[0]
        if world_size > 1:
            stats = torch.tensor(
                [sums[k] for k in METRIC_KEYS] + [float(n)], device=device
            )
            dist.all_reduce(stats)
            sums = {k: float(stats[i]) for i, k in enumerate(METRIC_KEYS)}
            n = float(stats[-1])
        results[name] = {k: v / max(n, 1) for k, v in sums.items()}
    raw_model.train()
    return results


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------

def _strip_frozen_encoder(state):
    """Drop frozen DINOv2 tensors from a state dict (reloadable from HF at
    inference; inference.py:load_model builds the encoder then loads with
    strict=False). Cuts ~700MB from each saved copy."""
    return {k: v for k, v in state.items() if not k.startswith("image_encoder.")}


def save_checkpoint(path, raw_model, ema_model, optimizer, cfg, norm_stats, step, best_val=None):
    """Save in the exact layout src/inference.py:load_model consumes.

    EMA weights are only saved once averaging has actually started
    (ema_start_step); before that the AveragedModel still holds the init copy
    and saving it as "ema" would silently publish untrained weights (the
    eval path defaults to EMA). ema=None makes loaders fall back to "model".
    """
    ema_started = ema_model is not None and int(ema_model.n_averaged.item()) > 0
    ckpt = {
        "model": _strip_frozen_encoder(raw_model.state_dict()),
        "ema": _strip_frozen_encoder(ema_model.state_dict()) if ema_started else None,
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "norm_stats": norm_stats,
        "weights_kind": "model",
        "step": step,
        "best_val": best_val,
        "val_metric": VAL_METRIC,  # metric semantics for best_val comparisons
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(ckpt, tmp)
    tmp.rename(path)


def load_pretrained(model, path, device):
    """Initialize `model` from HUG pretrained weights (safetensors or .pt).

    Unlike `train.resume` this only restores model weights - step/optimizer/
    best_val all start fresh (finetune setup). The released hug_full.safetensors
    stores EMA weights without the frozen DINOv2 image_encoder (loaded from HF
    at model build), so loading is strict=False and the loaded/missing counts
    are reported. Source normalization buffers are excluded because the target
    dataset's stats define the flow coordinate system. A true resume restores
    those buffers through the separate resume path.
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(p))
    else:
        sd = torch.load(str(p), map_location=device, weights_only=False)
        sd = sd.get("model", sd.get("ema", sd))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    ignored_norm_buffers = sorted(_PRETRAINED_NORM_BUFFER_KEYS.intersection(sd))
    if ignored_norm_buffers:
        sd = {k: v for k, v in sd.items() if k not in _PRETRAINED_NORM_BUFFER_KEYS}
        logging.info(
            "pretrained: kept target-dataset normalization and ignored %d "
            "source normalization buffers: %s",
            len(ignored_norm_buffers),
            ", ".join(ignored_norm_buffers),
        )

    incompatible, skipped = model.load_compatible_state_dict(sd)
    n_loaded = len(sd) - len(skipped)
    logging.info(
        f"pretrained <- {p}: loaded {n_loaded}/{len(sd)} source tensors "
        f"(missing = {len(incompatible.missing_keys)}, skipped = {len(skipped)})"
    )
    return n_loaded


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(
    config: Path,
    max_steps: Optional[int] = None,
    max_train_samples: Optional[int] = None,
) -> None:
    """Train HUG with DDP.

    Args:
        config: Path to the YAML config (see configs/train_hug.yaml).
        max_steps: Override trainer.train.total_steps (smoke tests).
        max_train_samples: Cap the train set size (smoke tests).
    """
    # Load config and initialize durable logs before DDP/data/model setup so
    # startup failures are persisted as well.
    cfg = OmegaConf.load(config)
    train_cfg = cfg.trainer.train
    if max_steps is not None:
        train_cfg.total_steps = max_steps
    out_dir = Path(train_cfg.output_dir)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log_path = _resolve_log_path(train_cfg, out_dir)
    writer, log_context = configure_logging(out_dir, log_path, rank, local_rank, world_size)
    tb_writer = None
    exception_state = {"step": None, "phase": "startup"}
    install_exception_hooks(log_context, exception_state)
    if writer:
        writer.write("startup", {"config": str(config), "output_dir": str(out_dir)})

    rank, local_rank, world_size, device = setup_ddp()
    torch.manual_seed(train_cfg.seed + rank)

    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
        console.print(f"[cyan]world_size={world_size}, output -> {out_dir}[/cyan]")
        if writer:
            writer.write("config_saved", {"config_copy": str(out_dir / "config.yaml")})

    logger.info("config loaded: %s", config, extra=log_context)
    if writer:
        writer.write("config_loaded", {"config": str(config)})
    norm_stats_file = Path(cfg.trainer.data.get("norm_stats_file", str(NORM_STATS_FILE)))
    with open(norm_stats_file) as f:
        norm_stats = json.load(f)
    logger.info("norm_stats loaded: %s", norm_stats_file, extra=log_context)
    if writer:
        writer.write("norm_stats_loaded", {"path": str(norm_stats_file)})
    if is_main(rank):
        console.print(f"[cyan]norm_stats: {norm_stats_file}[/cyan]")

    # ---- data ----
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loaders, sampler = build_loaders(
        cfg, train_ds, val_ds, rank, world_size, max_train_samples
    )
    logger.info("datasets ready: train=%d val=%d", len(train_ds), len(val_ds), extra=log_context)
    if writer:
        writer.write("datasets_ready", {"train_samples": len(train_ds), "val_samples": len(val_ds)})
    if is_main(rank):
        console.print(
            f"[cyan]train={len(train_ds)}  val={len(val_ds)}  "
            f"batch/GPU={train_cfg.batch_size}  global_batch={train_cfg.batch_size * world_size}[/cyan]"
        )

    # ---- model ----
    model = GraspFlowModel(cfg, norm_stats=norm_stats).to(device)
    logger.info("model initialized", extra=log_context)
    if writer:
        writer.write("model_initialized")
    if train_cfg.get("pretrained") and not train_cfg.get("resume"):
        # Finetune init: weights only, step/optimizer/best_val start fresh.
        # A resume checkpoint already contains the complete model state, so
        # skip the optional base pretrained load to avoid redundant I/O.
        load_pretrained(model, str(train_cfg.pretrained), device)
        logger.info("pretrained loaded: %s", train_cfg.pretrained, extra=log_context)
        if writer:
            writer.write("pretrained_loaded", {"path": str(train_cfg.pretrained)})
    model.train()
    ddp_model = (
        DDP(model, device_ids=[local_rank])
        if world_size > 1 and device.type == "cuda"
        else model
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg.lr,
        betas=tuple(train_cfg.betas),
        weight_decay=train_cfg.weight_decay,
    )

    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(train_cfg.ema_decay), use_buffers=False
    )

    # ---- resume ----
    start_step = 0
    best_val = float("inf")
    if train_cfg.get("resume"):
        ckpt = torch.load(train_cfg.resume, map_location=device, weights_only=False)
        # strict=False: checkpoints no longer store the frozen DINOv2 encoder
        model.load_compatible_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("ema") is not None:
            ema_state = ckpt["ema"]
            ema_model.module.load_compatible_state_dict(ema_state)
            if "n_averaged" in ema_state:
                ema_model.n_averaged.copy_(ema_state["n_averaged"].to(ema_model.n_averaged.device))
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val") or float("inf"))
        if ckpt.get("val_metric") != VAL_METRIC:
            # Old metric semantics (x0-recovery proxy ~2x optimistic): the
            # stored best_val is not comparable -> reset and re-baseline.
            logger.info(
                f"resume: val_metric {ckpt.get('val_metric')!r} != {VAL_METRIC!r} "
                f"-> resetting best_val (was {best_val:.4f})",
                extra=log_context,
            )
            best_val = float("inf")
        if is_main(rank):
            console.print(
                f"[yellow]resumed from {train_cfg.resume} @ step {start_step} "
                f"(best_val={best_val:.4f})[/yellow]"
            )

    if is_main(rank):
        tb_writer, tb_log_dir = _create_tensorboard_writer(
            train_cfg, out_dir, start_step, cfg=cfg
        )
        if tb_writer is not None:
            atexit.register(_close_tensorboard_writer, tb_writer)
            _safe_event(
                writer,
                logger,
                "tensorboard_ready",
                {"log_dir": str(tb_log_dir), "purge_after_step": start_step},
                start_step,
            )
            console.print(f"[cyan]tensorboard -> {tb_log_dir}[/cyan]")

    def lr_at(step: int) -> float:
        """Linear warmup -> cosine decay to lr * lr_min_ratio."""
        w = max(int(train_cfg.warmup_steps), 1)
        total = int(train_cfg.total_steps)
        min_ratio = float(train_cfg.get("lr_min_ratio", 0.0))
        if step < w:
            return train_cfg.lr * (step + 1) / w
        progress = min(max((step - w) / max(total - w, 1), 0.0), 1.0)
        return train_cfg.lr * (
            min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )

    # ---- train loop ----
    bf16 = bool(train_cfg.bf16) and device.type == "cuda"
    loader_iter = infinite_loader(train_loader, sampler)
    heartbeat_every = max(int(train_cfg.get("heartbeat_every", train_cfg.log_every)), 1)
    t_start = time.perf_counter()
    last_log_t, last_log_step = t_start, start_step
    _safe_event(
        writer,
        logger,
        "training_started",
        {"total_steps": int(train_cfg.total_steps), "start_step": start_step},
        start_step,
    )
    logger.info(
        "training loop started: step=%d total=%d",
        start_step,
        int(train_cfg.total_steps),
        extra=log_context,
    )

    for step in range(start_step, int(train_cfg.total_steps)):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        step_no = step + 1
        exception_state["step"] = step_no
        exception_state["phase"] = "batch"
        if step_no % heartbeat_every == 0:
            _safe_event(writer, logger, "heartbeat", {"phase": "batch"}, step_no)
        batch = next(loader_iter)
        exception_state["phase"] = "forward"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            preds, targets, time_weight = ddp_model(
                point_uv=batch["point_uv"].to(device, non_blocking=True),
                camera_K=batch["camera_K"].to(device, non_blocking=True),
                gt_mano_params=batch["mano_params"].to(device, non_blocking=True),
                rgb=batch["rgb"].to(device, non_blocking=True) if "rgb" in batch else None,
                pcl_xyz=(
                    batch["pcl_xyz"].to(device, non_blocking=True)
                    if "pcl_xyz" in batch
                    else None
                ),
                pcl_rgb=(
                    batch["pcl_rgb"].to(device, non_blocking=True)
                    if "pcl_rgb" in batch
                    else None
                ),
            )
        loss, comps = compute_loss(
            preds,
            targets,
            time_weight,
            lambda_v=float(train_cfg.lambda_v),
            lambda_3d=float(train_cfg.lambda_3d),
            lambda_mesh_3d=float(train_cfg.lambda_mesh_3d),
        )

        exception_state["phase"] = "backward"
        loss.backward()
        max_grad_norm = (
            float(train_cfg.grad_clip)
            if train_cfg.grad_clip and train_cfg.grad_clip > 0
            else float("inf")
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_grad_norm
        )
        comps["grad_norm"] = float(grad_norm.detach().cpu())
        optimizer.step()
        if step >= int(train_cfg.ema_start_step):
            ema_model.update_parameters(model)

        # ---- logging ----
        if is_main(rank) and step_no % int(train_cfg.log_every) == 0:
            now = time.perf_counter()
            sps = (step_no - last_log_step) / max(now - last_log_t, 1e-9)
            last_log_t, last_log_step = now, step_no
            comps.update({"step": step_no, "lr": lr, "samples_per_s": sps})
            console.print(
                f"step {step_no:>6d}  loss {comps['loss']:.4f}  "
                f"lv {comps['lv']:.4f}  l3d {comps['l3d']:.4f}  "
                f"lmesh {comps['lmesh']:.4f}  "
                f"lr {lr:.2e}  {sps:.1f} it/s"
            )
            _safe_event(writer, logger, "train", comps, step_no)
            _write_tensorboard_scalars(tb_writer, "train", comps, step_no)

        # ---- validation ----
        # Best-checkpoint score is the mean PA-MPJPE/PA-MPVPE (mm), averaged
        # equally across validation datasets. When EMA is active we validate the
        # EMA weights (what gets deployed/evaluated at test time), else the raw model.
        # All ranks evaluate their shard (rank-0-only would exceed the NCCL
        # watchdog timeout while other ranks wait at the barrier).
        if (step + 1) % int(train_cfg.val_every) == 0 or (step + 1) == int(
            train_cfg.total_steps
        ):
            ema_active = step + 1 >= int(train_cfg.ema_start_step)
            eval_model = ema_model.module if ema_active else model
            _phase_log(log_context, "validation", "start step=%d", step_no)
            _safe_event(writer, logger, "validation_started", {"weights": "ema" if ema_active else "model"}, step_no)
            exception_state["phase"] = "validation"
            val_results = run_val(eval_model, val_loaders, device, bf16, rank, world_size)
            exception_state["phase"] = "batch"
            _phase_log(log_context, "validation", "finished step=%d", step_no)
            ds_scores = {
                name: 0.5 * (r["pa_mpjpe"] + r["pa_mpvpe"])
                for name, r in val_results.items()
            }
            score = sum(ds_scores.values()) / len(ds_scores)
            if is_main(rank):
                console.print(f"[green]val @ {step + 1} ({'ema' if ema_active else 'model'}):[/green]")
                for name, r in val_results.items():
                    console.print(
                        f"  \\[{name}] MPJPE {r['mpjpe']:.2f} PA-MPJPE {r['pa_mpjpe']:.2f} "
                        f"MPVPE {r['mpvpe']:.2f} PA-MPVPE {r['pa_mpvpe']:.2f} mm"
                    )
                console.print(f"  [bold green]score {score:.2f}[/bold green]")
            _safe_event(
                writer,
                logger,
                "validation",
                {
                    "val": True,
                    "weights": "ema" if ema_active else "model",
                    "datasets": val_results,
                    "score": score,
                },
                step_no,
            )
            _write_tensorboard_scalars(
                tb_writer,
                "validation",
                {"datasets": val_results, "score": score},
                step_no,
            )
            if is_main(rank) and score < best_val:
                best_val = score
                ckpt_path = out_dir / "model_best.pt"
                exception_state["phase"] = "checkpoint_best"
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_started",
                    {"path": str(ckpt_path), "kind": "best"},
                    step_no,
                )
                try:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        ema_model,
                        optimizer,
                        cfg,
                        norm_stats,
                        step_no,
                        best_val=best_val,
                    )
                except BaseException:
                    logger.exception(
                        "checkpoint failed: path=%s step=%d",
                        ckpt_path,
                        step_no,
                        extra=log_context,
                    )
                    raise
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_finished",
                    {"path": str(ckpt_path), "kind": "best"},
                    step_no,
                )
                console.print(
                    f"[bold green]new best score {best_val:.4f} -> "
                    f"saved model_best.pt @ step {step_no}[/bold green]"
                )
                exception_state["phase"] = "batch"
            if world_size > 1:
                dist.barrier()

        if (step + 1) % int(train_cfg.ckpt_every) == 0 or (step + 1) == int(
            train_cfg.total_steps
        ):
            if is_main(rank):
                ckpt_path = out_dir / "model.pt"
                exception_state["phase"] = "checkpoint_periodic"
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_started",
                    {"path": str(ckpt_path), "kind": "periodic"},
                    step_no,
                )
                try:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        ema_model,
                        optimizer,
                        cfg,
                        norm_stats,
                        step_no,
                        best_val=best_val,
                    )
                except BaseException:
                    logger.exception(
                        "checkpoint failed: path=%s step=%d",
                        ckpt_path,
                        step_no,
                        extra=log_context,
                    )
                    raise
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_finished",
                    {"path": str(ckpt_path), "kind": "periodic"},
                    step_no,
                )
                console.print(f"[magenta]saved checkpoint @ step {step_no}[/magenta]")
                exception_state["phase"] = "batch"
            if world_size > 1:
                _phase_log(log_context, "barrier", "enter periodic checkpoint step=%d", step_no)
                dist.barrier()
                _phase_log(log_context, "barrier", "exit periodic checkpoint step=%d", step_no)

    _safe_event(writer, logger, "run_finished", {"status": "success"}, int(train_cfg.total_steps))
    _close_tensorboard_writer(tb_writer)
    atexit.unregister(_close_tensorboard_writer)
    logger.info("training finished successfully", extra=log_context)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except BaseException:
        # The installed hooks normally record this with the full traceback.
        # This fallback also guarantees stderr visibility if logging setup
        # itself failed before handlers could be installed.
        logger.exception("training process terminated", extra={"rank": os.environ.get("RANK", "0")})
        logging.shutdown()
        raise

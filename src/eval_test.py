"""Full-quantity evaluation on the official test splits (HO3D_v3 eval + DexYCB s0_test).

Two GT schemas are handled automatically per batch:

- DexYCB s0_test: converted pkls keep full MANO GT -> model.sample() +
  build_loss_dicts() -> MPJPE / PA-MPJPE / MPVPE / PA-MPVPE (mm).
- HO3D_v3 evaluation split: pkls carry joints_gt / verts_gt only (no MANO
  params in the official release) -> model.sample() -> mano_forward() ->
  same metrics against the stored GT.

Datasets come from the `trainer.test` section of the config; results are
printed per dataset and written to a JSON file (default
<output_dir>/test_results.json).

Usage (multi-GPU shards the sets):
    torchrun --nproc_per_node=4 -m src.eval_test --config configs/train_handrecon.yaml \
        [--ckpt /path/to/model_best.pt] [--weights ema|model] \
        [--sets dexycb_test,ho3d_eval] [--steps 50]
"""

import json
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Sampler

from .dataloader.grasp_dataset import GraspDataset
from .metrics import joint_mesh_errors
from .models.grasp_model import GraspFlowModel
from .models.native_mano import geometry_kwargs_from_batch
from .train import is_main, setup_ddp

console = Console()

METRIC_KEYS = ("mpjpe", "pa_mpjpe", "mpvpe", "pa_mpvpe")


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without DistributedSampler's tail padding."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def load_weights(model, ckpt, weights: str) -> None:
    """Load `model` or EMA weights from a train.py checkpoint.

    EMA state comes from torch's AveragedModel: strip the `module.` prefix and
    drop the `n_averaged` buffer. Checkpoints saved by train.py store no
    frozen image_encoder tensors (reloadable from HF) -> strict=False.
    """
    if weights == "ema" and ckpt.get("ema") is not None:
        sd = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in ckpt["ema"].items()
            if not k.startswith("n_averaged")
        }
    else:
        sd = ckpt["model"]
    incompatible, skipped = model.load_compatible_state_dict(sd)
    bad_missing = [k for k in incompatible.missing_keys if not k.startswith("image_encoder.")]
    # A 99D checkpoint loaded into a 109D model intentionally misses the new
    # shape heads; those are initialized from the constructor and fine-tuned.
    allowed_missing = {
        "flow.denoise_fn.proj_shape.weight",
        "flow.denoise_fn.out_shape.weight",
        "flow.denoise_fn.shape_mean",
        "flow.denoise_fn.shape_std",
    }
    legacy_without_skeleton = not any(
        key.startswith("fusion.skeleton_") for key in sd
    )
    bad_missing = [
        k
        for k in bad_missing
        if k not in allowed_missing
        and not (legacy_without_skeleton and k.startswith("fusion.skeleton_"))
    ]
    if bad_missing:
        raise RuntimeError(f"missing non-frozen keys: {bad_missing}")


def evaluate_dataset(
    name, model, loader, device, rank, world_size, bf16, steps=None
):
    """Run full-quantity evaluation over one dataset; returns {metric: mean_mm}."""
    model.eval()
    sums = {k: 0.0 for k in METRIC_KEYS}
    n = 0
    keypoint_samples = 0
    valid_keypoints = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                samples = model.sample(
                    point_uv=batch["point_uv"].to(device),
                    camera_K=batch["camera_K"].to(device),
                    rgb_camera_K=batch["rgb_camera_K"].to(device),
                    steps=steps,
                    rgb=batch["rgb"].to(device) if "rgb" in batch else None,
                    pcl_xyz=batch["pcl_xyz"].to(device) if "pcl_xyz" in batch else None,
                    pcl_rgb=batch["pcl_rgb"].to(device) if "pcl_rgb" in batch else None,
                    hand_keypoints_2d=batch["hand_keypoints_2d"].to(device),
                    hand_keypoints_valid=batch["hand_keypoints_valid"].to(device),
                    hand_keypoints_confidence=batch["hand_keypoints_confidence"].to(device),
                )
                if "mano_params" in batch:  # MANO GT available (DexYCB test)
                    preds, targets = model.build_loss_dicts(
                        samples, batch["mano_params"].to(device),
                        **geometry_kwargs_from_batch(batch, device),
                    )
                    errs = joint_mesh_errors(
                        preds["landmarks_3d"].float(),
                        targets["landmarks_3d"].float(),
                        preds["vertices"].float(),
                        targets["vertices"].float(),
                    )
                else:  # joints/verts GT only (HO3D_v3 evaluation split)
                    pred_out = model.mano_forward(samples)
                    errs = joint_mesh_errors(
                        pred_out["landmarks_3d"].float(),
                        batch["joints_gt"].to(device).float(),
                        pred_out["vertices"].float(),
                        batch["verts_gt"].to(device).float(),
                    )
            for k in METRIC_KEYS:
                sums[k] += errs[k].float().sum().item()
            n += errs["mpjpe"].shape[0]
            keypoint_samples += int(batch["has_hand_keypoints"].sum().item())
            valid_keypoints += int(batch["hand_keypoints_valid"].sum().item())
            if is_main(rank) and (i + 1) % 50 == 0:
                dt = time.perf_counter() - t0
                console.print(
                    f"  [{name}] batch {i + 1}/{len(loader)}, {n} samples, {dt:.0f}s"
                )

    # shard -> global sums/counts
    stats = torch.tensor(
        [sums[k] for k in METRIC_KEYS]
        + [float(n), float(keypoint_samples), float(valid_keypoints)],
        device=device,
    )
    if world_size > 1:
        torch.distributed.all_reduce(stats)
    global_n = max(stats[len(METRIC_KEYS)].item(), 1.0)
    means = {
        k: float(stats[i] / global_n)
        for i, k in enumerate(METRIC_KEYS)
    }
    means["n_samples"] = int(stats[len(METRIC_KEYS)].item())
    means["keypoint_sample_coverage"] = float(
        stats[len(METRIC_KEYS) + 1] / global_n
    )
    means["mean_valid_keypoints"] = float(
        stats[len(METRIC_KEYS) + 2] / global_n
    )
    means["seconds"] = round(time.perf_counter() - t0, 1)
    return means


def main(
    config: Path,
    ckpt: Optional[Path] = None,
    weights: str = "ema",
    sets: Optional[str] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    """Evaluate a checkpoint on the official test splits, full quantity.

    Args:
        config: training config YAML (model + data + test sections).
        ckpt: checkpoint to evaluate; defaults to <output_dir>/model_best.pt.
        weights: "ema" (default) or "model".
        sets: comma-separated dataset names to evaluate (default: all in cfg).
        steps: flow sampling steps override (default: cfg sampling_steps).
        batch_size: per-GPU batch size override (default: trainer.test.batch_size).
        limit: cap samples per dataset (smoke tests; NOT for real results).
    """
    rank, local_rank, world_size, device = setup_ddp()
    torch.manual_seed(42)

    cfg = OmegaConf.load(config)
    train_cfg = cfg.trainer.train
    test_cfg = cfg.trainer.get("test")
    assert test_cfg is not None and test_cfg.get("datasets"), (
        "config has no trainer.test.datasets section - see configs/train_handrecon.yaml"
    )

    ckpt_path = Path(ckpt) if ckpt else Path(train_cfg.output_dir) / "model_best.pt"
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats = loaded.get("norm_stats")
    if norm_stats is None:
        norm_stats_file = Path(cfg.trainer.data.norm_stats_file)
        with open(norm_stats_file) as f:
            norm_stats = json.load(f)
    model = GraspFlowModel(cfg, norm_stats=norm_stats).to(device)
    load_weights(model, loaded, weights)
    model.eval()
    if is_main(rank):
        console.print(
            f"[cyan]ckpt={ckpt_path} weights={weights} world_size={world_size}[/cyan]"
        )

    common = dict(
        split="val",
        n_points_input=cfg.trainer.data.n_points_input,
        pcl_crop_radius=cfg.trainer.model.get("pcl_crop_radius", 0.3),
        d_mano=int(cfg.trainer.model.get("d_mano", 99)),
        use_rgb=cfg.trainer.model.get("use_rgb", True),
        use_depth=cfg.trainer.model.get("use_depth", True),
        hand_crop=cfg.trainer.data.get("hand_crop", {}),
        geometry_overlay=cfg.trainer.data.get("geometry_overlay"),
    )
    bs = int(batch_size or test_cfg.get("batch_size", 256))
    bf16 = bool(train_cfg.get("bf16", True)) and device.type == "cuda"

    wanted = set(sets.split(",")) if sets else None
    results = {}
    for entry in test_cfg.datasets:
        if wanted is not None and entry.name not in wanted:
            continue
        ds = GraspDataset(
            str(entry.path),
            samples_filename=str(entry.samples) if entry.get("samples") else None,
            **common,
        )
        if limit is not None:
            ds.grasp_files = ds.grasp_files[: int(limit)]
        sampler = (
            DistributedEvalSampler(ds, num_replicas=world_size, rank=rank)
            if world_size > 1
            else None
        )
        loader = DataLoader(
            ds,
            batch_size=bs,
            sampler=sampler,
            shuffle=False,
            num_workers=(
                0
                if bool(cfg.trainer.data.get("hand_crop", {}).get("enabled", False))
                else cfg.trainer.data.num_workers
            ),
            pin_memory=True,
        )
        if is_main(rank):
            console.print(f"[cyan]== {entry.name}: {len(ds)} samples ==[/cyan]")
        results[entry.name] = evaluate_dataset(
            entry.name, model, loader, device, rank, world_size, bf16, steps=steps
        )
        if world_size > 1:
            torch.distributed.barrier()

    if is_main(rank):
        table = Table(title="Official test-set evaluation (mm)")
        table.add_column("dataset")
        table.add_column("n", justify="right")
        for k in METRIC_KEYS:
            table.add_column(k.upper(), justify="right")
        table.add_column("KP COVER", justify="right")
        table.add_column("VALID KP", justify="right")
        for name, r in results.items():
            table.add_row(
                name, str(r["n_samples"]),
                *[f"{r[k]:.2f}" for k in METRIC_KEYS],
                f"{100.0 * r['keypoint_sample_coverage']:.2f}%",
                f"{r['mean_valid_keypoints']:.2f}",
            )
        console.print(table)

        out_path = Path(
            test_cfg.get("out") or (Path(train_cfg.output_dir) / "test_results.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config),
            "ckpt": str(ckpt_path),
            "weights": weights,
            "step": loaded.get("step"),
            "sampling_steps": int(
                steps or cfg.trainer.model.get("sampling_steps", 50)
            ),
            "eval_keypoint_source": str(
                cfg.trainer.data.get("hand_crop", {}).get(
                    "eval_keypoint_source",
                    cfg.trainer.data.get("hand_crop", {}).get(
                        "keypoint_source", "mediapipe"
                    ),
                )
            ),
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"[green]wrote {out_path}[/green]")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
